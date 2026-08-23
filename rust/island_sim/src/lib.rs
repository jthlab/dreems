use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

fn build_adjacency(
    num_demes: usize,
    edges: ArrayView2<'_, i32>,
    rates: &[f64],
    symmetric: bool,
) -> PyResult<(Vec<Vec<(usize, f64)>>, Vec<f64>)> {
    if edges.shape()[1] != 2 || edges.shape()[0] != rates.len() {
        return Err(PyValueError::new_err(
            "migration_edges must have shape (num_edges, 2) and rates must match",
        ));
    }

    let mut adjacency = vec![Vec::<(usize, f64)>::new(); num_demes];
    for edge_index in 0..edges.shape()[0] {
        let left = edges[[edge_index, 0]];
        let right = edges[[edge_index, 1]];
        let rate = rates[edge_index];
        if left < 0 || right < 0 || left as usize >= num_demes || right as usize >= num_demes {
            return Err(PyValueError::new_err(
                "migration_edges entries must be in [0, num_demes)",
            ));
        }
        if rate < 0.0 || !rate.is_finite() {
            return Err(PyValueError::new_err(
                "migration_rates must contain finite non-negative values",
            ));
        }
        if rate == 0.0 || left == right {
            continue;
        }
        adjacency[left as usize].push((right as usize, rate));
        if symmetric {
            adjacency[right as usize].push((left as usize, rate));
        }
    }
    let outgoing_totals = adjacency
        .iter()
        .map(|outgoing| outgoing.iter().map(|(_, rate)| *rate).sum())
        .collect::<Vec<f64>>();
    Ok((adjacency, outgoing_totals))
}

fn add_occupied(occupied: &mut Vec<usize>, occupied_pos: &mut [usize], deme: usize) {
    if occupied_pos[deme] == usize::MAX {
        occupied_pos[deme] = occupied.len();
        occupied.push(deme);
    }
}

fn remove_occupied(occupied: &mut Vec<usize>, occupied_pos: &mut [usize], deme: usize) {
    let pos = occupied_pos[deme];
    if pos == usize::MAX {
        return;
    }
    let last = occupied
        .pop()
        .expect("occupied deme list must not be empty");
    if pos < occupied.len() {
        occupied[pos] = last;
        occupied_pos[last] = pos;
    }
    occupied_pos[deme] = usize::MAX;
}

fn weighted_outgoing(rng: &mut SmallRng, outgoing: &[(usize, f64)], total: f64) -> Option<usize> {
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    let mut draw = rng.gen::<f64>() * total;
    for (target, weight) in outgoing {
        if *weight <= 0.0 {
            continue;
        }
        if draw < *weight {
            return Some(*target);
        }
        draw -= *weight;
    }
    outgoing
        .iter()
        .rfind(|(_, weight)| *weight > 0.0)
        .map(|(target, _)| *target)
}

fn weighted_coalescence_deme(
    rng: &mut SmallRng,
    occupied: &[usize],
    deme_lineages: &[Vec<usize>],
    q: &[f64],
    total: f64,
) -> Option<usize> {
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    let mut draw = rng.gen::<f64>() * total;
    let mut fallback = None;
    for deme in occupied {
        let k = deme_lineages[*deme].len() as f64;
        let weight = q[*deme] * k * (k - 1.0) * 0.5;
        if weight <= 0.0 {
            continue;
        }
        fallback = Some(*deme);
        if draw < weight {
            return Some(*deme);
        }
        draw -= weight;
    }
    fallback
}

fn weighted_migration_deme(
    rng: &mut SmallRng,
    occupied: &[usize],
    deme_lineages: &[Vec<usize>],
    outgoing_totals: &[f64],
    total: f64,
) -> Option<usize> {
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    let mut draw = rng.gen::<f64>() * total;
    let mut fallback = None;
    for deme in occupied {
        let weight = deme_lineages[*deme].len() as f64 * outgoing_totals[*deme];
        if weight <= 0.0 {
            continue;
        }
        fallback = Some(*deme);
        if draw < weight {
            return Some(*deme);
        }
        draw -= weight;
    }
    fallback
}

fn random_exponential_time(rng: &mut SmallRng, rate: f64) -> Option<f64> {
    if rate <= 0.0 || !rate.is_finite() {
        return None;
    }
    let u = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    Some(-u.ln() / rate)
}

fn counts_pass_allele_frequency(
    counts: &[u32],
    sample_nodes: &[i32],
    ploidy: usize,
    min_allele_frequency: f64,
    max_allele_frequency: f64,
) -> bool {
    let total_haploids = sample_nodes.len().saturating_mul(ploidy);
    if total_haploids == 0 {
        return false;
    }
    let derived = counts.iter().map(|value| *value as usize).sum::<usize>();
    let frequency = derived as f64 / total_haploids as f64;
    frequency >= min_allele_frequency && frequency <= max_allele_frequency
}

fn counts_from_bitmask(mask: u128, haploid_demes: &[usize], num_demes: usize) -> Vec<u32> {
    let mut counts = vec![0_u32; num_demes];
    for (haploid, deme) in haploid_demes.iter().enumerate() {
        if ((mask >> haploid) & 1_u128) == 1 {
            counts[*deme] += 1;
        }
    }
    counts
}

fn counts_from_words(mask: &[u128], haploid_demes: &[usize], num_demes: usize) -> Vec<u32> {
    let mut counts = vec![0_u32; num_demes];
    for (word_index, word) in mask.iter().copied().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let haploid = word_index * 128 + bit;
            if haploid < haploid_demes.len() {
                counts[haploid_demes[haploid]] += 1;
            }
            remaining &= remaining - 1;
        }
    }
    counts
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MarkerModel {
    FirstMutation,
    RandomMarker,
    ConditionedRandomMarker,
}

fn parse_marker_model(marker_model: &str) -> PyResult<MarkerModel> {
    match marker_model {
        "first_mutation" | "first" => Ok(MarkerModel::FirstMutation),
        "random_marker" | "uniform_marker" | "feems_paper" => Ok(MarkerModel::RandomMarker),
        "branch_marker"
        | "conditioned_first_mutation"
        | "conditioned_marker"
        | "conditioned_random_marker"
        | "segregating_marker" => Ok(MarkerModel::ConditionedRandomMarker),
        _ => Err(PyValueError::new_err(
            "marker_model must be first_mutation, random_marker, or conditioned_random_marker",
        )),
    }
}

fn validate_allele_frequency_filter(
    min_allele_frequency: f64,
    max_allele_frequency: f64,
) -> PyResult<()> {
    if !min_allele_frequency.is_finite()
        || !max_allele_frequency.is_finite()
        || min_allele_frequency < 0.0
        || max_allele_frequency > 1.0
        || min_allele_frequency > max_allele_frequency
    {
        return Err(PyValueError::new_err(
            "allele-frequency thresholds must satisfy 0 <= min <= max <= 1",
        ));
    }
    Ok(())
}

fn remove_active(active: &mut Vec<usize>, active_pos: &mut [usize], lineage: usize) {
    let pos = active_pos[lineage];
    let last = active.pop().expect("active lineage list must not be empty");
    if pos < active.len() {
        active[pos] = last;
        active_pos[last] = pos;
    }
    active_pos[lineage] = usize::MAX;
}

fn simulate_one_counts_words(
    rng: &mut SmallRng,
    sample_nodes: &[i32],
    num_demes: usize,
    q: &[f64],
    adjacency: &[Vec<(usize, f64)>],
    outgoing_totals: &[f64],
    ploidy: usize,
    mutation_rate: f64,
) -> Option<Vec<u32>> {
    let initial_haploids = sample_nodes.len().checked_mul(ploidy)?;
    let word_count = (initial_haploids + 127) / 128;
    let mut deme_lineages = vec![Vec::<usize>::new(); num_demes];
    let mut lineage_deme = Vec::<usize>::new();
    let mut descendants = vec![0_u128; initial_haploids.checked_mul(word_count)?];
    let mut active = Vec::<usize>::new();
    let mut active_pos = Vec::<usize>::new();
    let mut occupied = Vec::<usize>::new();
    let mut occupied_pos = vec![usize::MAX; num_demes];
    let mut haploid_demes = Vec::<usize>::new();

    for sample_deme in sample_nodes {
        let deme = *sample_deme as usize;
        for _copy in 0..ploidy {
            let lineage = lineage_deme.len();
            lineage_deme.push(deme);
            descendants[lineage * word_count + lineage / 128] = 1_u128 << (lineage % 128);
            deme_lineages[deme].push(lineage);
            active_pos.push(active.len());
            active.push(lineage);
            haploid_demes.push(deme);
        }
        add_occupied(&mut occupied, &mut occupied_pos, deme);
    }

    let total_haploids = active.len() as u32;
    if total_haploids < 2 {
        return None;
    }

    while active.len() > 1 {
        let mut total_coalescence = 0.0_f64;
        let mut total_migration = 0.0_f64;
        for deme in &occupied {
            let k = deme_lineages[*deme].len() as f64;
            total_coalescence += q[*deme] * k * (k - 1.0) * 0.5;
            total_migration += k * outgoing_totals[*deme];
        }
        let total_mutation = mutation_rate * active.len() as f64;
        let total_rate = total_coalescence + total_migration + total_mutation;
        if total_rate <= 0.0 || !total_rate.is_finite() {
            return None;
        }

        let mut event = rng.gen::<f64>() * total_rate;
        if event < total_mutation {
            let lineage = active[rng.gen_range(0..active.len())];
            let start = lineage * word_count;
            let end = start + word_count;
            let mask = &descendants[start..end];
            let total: u32 = mask.iter().map(|word| word.count_ones()).sum();
            if total == 0 || total >= total_haploids {
                return None;
            }
            let mut counts = vec![0_u32; num_demes];
            for (word_index, word) in mask.iter().copied().enumerate() {
                let mut remaining = word;
                while remaining != 0 {
                    let bit = remaining.trailing_zeros() as usize;
                    let haploid = word_index * 128 + bit;
                    if haploid < haploid_demes.len() {
                        counts[haploid_demes[haploid]] += 1;
                    }
                    remaining &= remaining - 1;
                }
            }
            return Some(counts);
        }
        event -= total_mutation;

        if event < total_coalescence {
            let deme =
                weighted_coalescence_deme(rng, &occupied, &deme_lineages, q, total_coalescence)?;
            let lineages = &mut deme_lineages[deme];
            if lineages.len() < 2 {
                return None;
            }
            let first_pos = rng.gen_range(0..lineages.len());
            let mut second_pos = rng.gen_range(0..lineages.len() - 1);
            if second_pos >= first_pos {
                second_pos += 1;
            }
            let (keep_pos, remove_pos) = if first_pos < second_pos {
                (first_pos, second_pos)
            } else {
                (second_pos, first_pos)
            };
            let keep = lineages[keep_pos];
            let remove = lineages[remove_pos];
            lineages.remove(remove_pos);
            let keep_start = keep * word_count;
            let remove_start = remove * word_count;
            for word in 0..word_count {
                let remove_value = descendants[remove_start + word];
                descendants[keep_start + word] |= remove_value;
                descendants[remove_start + word] = 0;
            }
            lineage_deme[remove] = usize::MAX;
            remove_active(&mut active, &mut active_pos, remove);
            continue;
        }

        let deme = weighted_migration_deme(
            rng,
            &occupied,
            &deme_lineages,
            outgoing_totals,
            total_migration,
        )?;
        if deme_lineages[deme].is_empty() || adjacency[deme].is_empty() {
            return None;
        }
        let lineage_pos = rng.gen_range(0..deme_lineages[deme].len());
        let lineage = deme_lineages[deme].swap_remove(lineage_pos);
        if deme_lineages[deme].is_empty() {
            remove_occupied(&mut occupied, &mut occupied_pos, deme);
        }
        let target = weighted_outgoing(rng, &adjacency[deme], outgoing_totals[deme])?;
        lineage_deme[lineage] = target;
        if deme_lineages[target].is_empty() {
            add_occupied(&mut occupied, &mut occupied_pos, target);
        }
        deme_lineages[target].push(lineage);
    }
    None
}

fn simulate_one_counts_bitmask(
    rng: &mut SmallRng,
    sample_nodes: &[i32],
    num_demes: usize,
    q: &[f64],
    adjacency: &[Vec<(usize, f64)>],
    outgoing_totals: &[f64],
    ploidy: usize,
    mutation_rate: f64,
) -> Option<Vec<u32>> {
    let mut deme_lineages = vec![Vec::<usize>::new(); num_demes];
    let mut lineage_deme = Vec::<usize>::new();
    let mut descendants = Vec::<u128>::new();
    let mut active = Vec::<usize>::new();
    let mut active_pos = Vec::<usize>::new();
    let mut occupied = Vec::<usize>::new();
    let mut occupied_pos = vec![usize::MAX; num_demes];
    let mut haploid_demes = Vec::<usize>::new();

    for sample_deme in sample_nodes {
        let deme = *sample_deme as usize;
        for _copy in 0..ploidy {
            let lineage = lineage_deme.len();
            if lineage >= 128 {
                return None;
            }
            lineage_deme.push(deme);
            descendants.push(1_u128 << lineage);
            deme_lineages[deme].push(lineage);
            active_pos.push(active.len());
            active.push(lineage);
            haploid_demes.push(deme);
        }
        add_occupied(&mut occupied, &mut occupied_pos, deme);
    }

    let total_haploids = active.len() as u32;
    if total_haploids < 2 {
        return None;
    }

    while active.len() > 1 {
        let mut total_coalescence = 0.0_f64;
        let mut total_migration = 0.0_f64;
        for deme in &occupied {
            let k = deme_lineages[*deme].len() as f64;
            total_coalescence += q[*deme] * k * (k - 1.0) * 0.5;
            total_migration += k * outgoing_totals[*deme];
        }
        let total_mutation = mutation_rate * active.len() as f64;
        let total_rate = total_coalescence + total_migration + total_mutation;
        if total_rate <= 0.0 || !total_rate.is_finite() {
            return None;
        }

        let mut event = rng.gen::<f64>() * total_rate;
        if event < total_mutation {
            let lineage = active[rng.gen_range(0..active.len())];
            let mask = descendants[lineage];
            let total = mask.count_ones();
            if total == 0 || total >= total_haploids {
                return None;
            }
            let mut counts = vec![0_u32; num_demes];
            for (haploid, deme) in haploid_demes.iter().enumerate() {
                if ((mask >> haploid) & 1_u128) == 1 {
                    counts[*deme] += 1;
                }
            }
            return Some(counts);
        }
        event -= total_mutation;

        if event < total_coalescence {
            let deme =
                weighted_coalescence_deme(rng, &occupied, &deme_lineages, q, total_coalescence)?;
            let lineages = &mut deme_lineages[deme];
            if lineages.len() < 2 {
                return None;
            }
            let first_pos = rng.gen_range(0..lineages.len());
            let mut second_pos = rng.gen_range(0..lineages.len() - 1);
            if second_pos >= first_pos {
                second_pos += 1;
            }
            let (keep_pos, remove_pos) = if first_pos < second_pos {
                (first_pos, second_pos)
            } else {
                (second_pos, first_pos)
            };
            let keep = lineages[keep_pos];
            let remove = lineages[remove_pos];
            lineages.remove(remove_pos);
            descendants[keep] |= descendants[remove];
            descendants[remove] = 0;
            lineage_deme[remove] = usize::MAX;
            remove_active(&mut active, &mut active_pos, remove);
            continue;
        }

        let deme = weighted_migration_deme(
            rng,
            &occupied,
            &deme_lineages,
            outgoing_totals,
            total_migration,
        )?;
        if deme_lineages[deme].is_empty() || adjacency[deme].is_empty() {
            return None;
        }
        let lineage_pos = rng.gen_range(0..deme_lineages[deme].len());
        let lineage = deme_lineages[deme].swap_remove(lineage_pos);
        if deme_lineages[deme].is_empty() {
            remove_occupied(&mut occupied, &mut occupied_pos, deme);
        }
        let target = weighted_outgoing(rng, &adjacency[deme], outgoing_totals[deme])?;
        lineage_deme[lineage] = target;
        if deme_lineages[target].is_empty() {
            add_occupied(&mut occupied, &mut occupied_pos, target);
        }
        deme_lineages[target].push(lineage);
    }
    None
}

fn simulate_one_counts_random_marker_bitmask(
    rng: &mut SmallRng,
    sample_nodes: &[i32],
    num_demes: usize,
    q: &[f64],
    adjacency: &[Vec<(usize, f64)>],
    outgoing_totals: &[f64],
    ploidy: usize,
    mutation_rate: f64,
    condition_on_mutation: bool,
) -> Option<Vec<u32>> {
    let mut deme_lineages = vec![Vec::<usize>::new(); num_demes];
    let mut lineage_deme = Vec::<usize>::new();
    let mut descendants = Vec::<u128>::new();
    let mut active = Vec::<usize>::new();
    let mut active_pos = Vec::<usize>::new();
    let mut occupied = Vec::<usize>::new();
    let mut occupied_pos = vec![usize::MAX; num_demes];
    let mut haploid_demes = Vec::<usize>::new();

    for sample_deme in sample_nodes {
        let deme = *sample_deme as usize;
        for _copy in 0..ploidy {
            let lineage = lineage_deme.len();
            if lineage >= 128 {
                return None;
            }
            lineage_deme.push(deme);
            descendants.push(1_u128 << lineage);
            deme_lineages[deme].push(lineage);
            active_pos.push(active.len());
            active.push(lineage);
            haploid_demes.push(deme);
        }
        add_occupied(&mut occupied, &mut occupied_pos, deme);
    }

    if active.len() < 2 {
        return None;
    }

    let mut branch_mass = 0.0_f64;
    let mut selected_mask = 0_u128;
    let mut has_selected = false;

    while active.len() > 1 {
        let mut total_coalescence = 0.0_f64;
        let mut total_migration = 0.0_f64;
        for deme in &occupied {
            let k = deme_lineages[*deme].len() as f64;
            total_coalescence += q[*deme] * k * (k - 1.0) * 0.5;
            total_migration += k * outgoing_totals[*deme];
        }
        let total_rate = total_coalescence + total_migration;
        let duration = random_exponential_time(rng, total_rate)?;
        let interval_mass = duration * active.len() as f64;
        if interval_mass > 0.0 && interval_mass.is_finite() {
            let next_mass = branch_mass + interval_mass;
            if rng.gen::<f64>() * next_mass < interval_mass {
                let lineage = active[rng.gen_range(0..active.len())];
                selected_mask = descendants[lineage];
                has_selected = true;
            }
            branch_mass = next_mass;
        }

        let event = rng.gen::<f64>() * total_rate;
        if event < total_coalescence {
            let deme =
                weighted_coalescence_deme(rng, &occupied, &deme_lineages, q, total_coalescence)?;
            let lineages = &mut deme_lineages[deme];
            if lineages.len() < 2 {
                return None;
            }
            let first_pos = rng.gen_range(0..lineages.len());
            let mut second_pos = rng.gen_range(0..lineages.len() - 1);
            if second_pos >= first_pos {
                second_pos += 1;
            }
            let (keep_pos, remove_pos) = if first_pos < second_pos {
                (first_pos, second_pos)
            } else {
                (second_pos, first_pos)
            };
            let keep = lineages[keep_pos];
            let remove = lineages[remove_pos];
            lineages.remove(remove_pos);
            descendants[keep] |= descendants[remove];
            descendants[remove] = 0;
            lineage_deme[remove] = usize::MAX;
            remove_active(&mut active, &mut active_pos, remove);
            continue;
        }

        let deme = weighted_migration_deme(
            rng,
            &occupied,
            &deme_lineages,
            outgoing_totals,
            total_migration,
        )?;
        if deme_lineages[deme].is_empty() || adjacency[deme].is_empty() {
            return None;
        }
        let lineage_pos = rng.gen_range(0..deme_lineages[deme].len());
        let lineage = deme_lineages[deme].swap_remove(lineage_pos);
        if deme_lineages[deme].is_empty() {
            remove_occupied(&mut occupied, &mut occupied_pos, deme);
        }
        let target = weighted_outgoing(rng, &adjacency[deme], outgoing_totals[deme])?;
        lineage_deme[lineage] = target;
        if deme_lineages[target].is_empty() {
            add_occupied(&mut occupied, &mut occupied_pos, target);
        }
        deme_lineages[target].push(lineage);
    }

    if !has_selected || branch_mass <= 0.0 || !branch_mass.is_finite() {
        return None;
    }
    if !condition_on_mutation {
        let mutation_probability = 1.0 - (-mutation_rate * branch_mass).exp();
        if rng.gen::<f64>() >= mutation_probability {
            return None;
        }
    }
    Some(counts_from_bitmask(
        selected_mask,
        &haploid_demes,
        num_demes,
    ))
}

fn simulate_one_counts_random_marker_words(
    rng: &mut SmallRng,
    sample_nodes: &[i32],
    num_demes: usize,
    q: &[f64],
    adjacency: &[Vec<(usize, f64)>],
    outgoing_totals: &[f64],
    ploidy: usize,
    mutation_rate: f64,
    condition_on_mutation: bool,
) -> Option<Vec<u32>> {
    let initial_haploids = sample_nodes.len().checked_mul(ploidy)?;
    let word_count = (initial_haploids + 127) / 128;
    let mut deme_lineages = vec![Vec::<usize>::new(); num_demes];
    let mut lineage_deme = Vec::<usize>::new();
    let mut descendants = vec![0_u128; initial_haploids.checked_mul(word_count)?];
    let mut active = Vec::<usize>::new();
    let mut active_pos = Vec::<usize>::new();
    let mut occupied = Vec::<usize>::new();
    let mut occupied_pos = vec![usize::MAX; num_demes];
    let mut haploid_demes = Vec::<usize>::new();

    for sample_deme in sample_nodes {
        let deme = *sample_deme as usize;
        for _copy in 0..ploidy {
            let lineage = lineage_deme.len();
            lineage_deme.push(deme);
            descendants[lineage * word_count + lineage / 128] = 1_u128 << (lineage % 128);
            deme_lineages[deme].push(lineage);
            active_pos.push(active.len());
            active.push(lineage);
            haploid_demes.push(deme);
        }
        add_occupied(&mut occupied, &mut occupied_pos, deme);
    }

    if active.len() < 2 {
        return None;
    }

    let mut branch_mass = 0.0_f64;
    let mut selected_mask = vec![0_u128; word_count];
    let mut has_selected = false;

    while active.len() > 1 {
        let mut total_coalescence = 0.0_f64;
        let mut total_migration = 0.0_f64;
        for deme in &occupied {
            let k = deme_lineages[*deme].len() as f64;
            total_coalescence += q[*deme] * k * (k - 1.0) * 0.5;
            total_migration += k * outgoing_totals[*deme];
        }
        let total_rate = total_coalescence + total_migration;
        let duration = random_exponential_time(rng, total_rate)?;
        let interval_mass = duration * active.len() as f64;
        if interval_mass > 0.0 && interval_mass.is_finite() {
            let next_mass = branch_mass + interval_mass;
            if rng.gen::<f64>() * next_mass < interval_mass {
                let lineage = active[rng.gen_range(0..active.len())];
                let start = lineage * word_count;
                let end = start + word_count;
                selected_mask.copy_from_slice(&descendants[start..end]);
                has_selected = true;
            }
            branch_mass = next_mass;
        }

        let event = rng.gen::<f64>() * total_rate;
        if event < total_coalescence {
            let deme =
                weighted_coalescence_deme(rng, &occupied, &deme_lineages, q, total_coalescence)?;
            let lineages = &mut deme_lineages[deme];
            if lineages.len() < 2 {
                return None;
            }
            let first_pos = rng.gen_range(0..lineages.len());
            let mut second_pos = rng.gen_range(0..lineages.len() - 1);
            if second_pos >= first_pos {
                second_pos += 1;
            }
            let (keep_pos, remove_pos) = if first_pos < second_pos {
                (first_pos, second_pos)
            } else {
                (second_pos, first_pos)
            };
            let keep = lineages[keep_pos];
            let remove = lineages[remove_pos];
            lineages.remove(remove_pos);
            let keep_start = keep * word_count;
            let remove_start = remove * word_count;
            for word in 0..word_count {
                let remove_value = descendants[remove_start + word];
                descendants[keep_start + word] |= remove_value;
                descendants[remove_start + word] = 0;
            }
            lineage_deme[remove] = usize::MAX;
            remove_active(&mut active, &mut active_pos, remove);
            continue;
        }

        let deme = weighted_migration_deme(
            rng,
            &occupied,
            &deme_lineages,
            outgoing_totals,
            total_migration,
        )?;
        if deme_lineages[deme].is_empty() || adjacency[deme].is_empty() {
            return None;
        }
        let lineage_pos = rng.gen_range(0..deme_lineages[deme].len());
        let lineage = deme_lineages[deme].swap_remove(lineage_pos);
        if deme_lineages[deme].is_empty() {
            remove_occupied(&mut occupied, &mut occupied_pos, deme);
        }
        let target = weighted_outgoing(rng, &adjacency[deme], outgoing_totals[deme])?;
        lineage_deme[lineage] = target;
        if deme_lineages[target].is_empty() {
            add_occupied(&mut occupied, &mut occupied_pos, target);
        }
        deme_lineages[target].push(lineage);
    }

    if !has_selected || branch_mass <= 0.0 || !branch_mass.is_finite() {
        return None;
    }
    if !condition_on_mutation {
        let mutation_probability = 1.0 - (-mutation_rate * branch_mass).exp();
        if rng.gen::<f64>() >= mutation_probability {
            return None;
        }
    }
    Some(counts_from_words(&selected_mask, &haploid_demes, num_demes))
}

fn counts_to_genotypes(
    rng: &mut SmallRng,
    allele_counts: &[u32],
    num_snps: usize,
    num_demes: usize,
    sample_nodes: &[i32],
    ploidy: usize,
) -> PyResult<Vec<u8>> {
    let mut samples_by_deme = vec![Vec::<usize>::new(); num_demes];
    for (sample_index, deme) in sample_nodes.iter().enumerate() {
        samples_by_deme[*deme as usize].push(sample_index);
    }
    let sampled_demes = samples_by_deme
        .iter()
        .enumerate()
        .filter_map(
            |(deme, samples)| {
                if samples.is_empty() {
                    None
                } else {
                    Some(deme)
                }
            },
        )
        .collect::<Vec<usize>>();
    let mut genotypes = vec![0_u8; sample_nodes.len() * num_snps];
    for snp in 0..num_snps {
        for deme in &sampled_demes {
            let sample_indices = &samples_by_deme[*deme];
            let haploid_count = sample_indices.len() * ploidy;
            let derived = allele_counts[snp * num_demes + *deme] as usize;
            if derived > haploid_count {
                return Err(PyValueError::new_err(
                    "allele counts are inconsistent with sample_nodes and ploidy",
                ));
            }
            let mut haplotypes = vec![0_u8; haploid_count];
            for value in haplotypes.iter_mut().take(derived) {
                *value = 1;
            }
            haplotypes.shuffle(rng);
            for (local, sample_index) in sample_indices.iter().enumerate() {
                let start = local * ploidy;
                let count: u8 = haplotypes[start..start + ploidy].iter().sum();
                genotypes[*sample_index * num_snps + snp] = count;
            }
        }
    }
    Ok(genotypes)
}

fn sample_layout(sample_nodes: &[i32], num_demes: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut samples_by_deme = vec![Vec::<usize>::new(); num_demes];
    for (sample_index, deme) in sample_nodes.iter().enumerate() {
        samples_by_deme[*deme as usize].push(sample_index);
    }
    let sampled_demes = samples_by_deme
        .iter()
        .enumerate()
        .filter_map(
            |(deme, samples)| {
                if samples.is_empty() {
                    None
                } else {
                    Some(deme)
                }
            },
        )
        .collect::<Vec<usize>>();
    (samples_by_deme, sampled_demes)
}

fn accumulate_covariance_from_counts(
    rng: &mut SmallRng,
    counts: &[u32],
    samples_by_deme: &[Vec<usize>],
    sampled_demes: &[usize],
    ploidy: usize,
    genotype_column: &mut [f64],
    covariance_sum: &mut [f64],
) -> PyResult<bool> {
    genotype_column.fill(0.0);
    for deme in sampled_demes {
        let sample_indices = &samples_by_deme[*deme];
        let haploid_count = sample_indices.len() * ploidy;
        let derived = counts[*deme] as usize;
        if derived > haploid_count {
            return Err(PyValueError::new_err(
                "allele counts are inconsistent with sample_nodes and ploidy",
            ));
        }
        let mut haplotypes = vec![0_u8; haploid_count];
        for value in haplotypes.iter_mut().take(derived) {
            *value = 1;
        }
        haplotypes.shuffle(rng);
        for (local, sample_index) in sample_indices.iter().enumerate() {
            let start = local * ploidy;
            let count: u8 = haplotypes[start..start + ploidy].iter().sum();
            genotype_column[*sample_index] = f64::from(count);
        }
    }

    let sample_count = genotype_column.len();
    if sample_count < 2 {
        return Ok(false);
    }
    let mean = genotype_column.iter().sum::<f64>() / sample_count as f64;
    let centered_sum_squares = genotype_column
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>();
    if centered_sum_squares <= 1e-16 || !centered_sum_squares.is_finite() {
        return Ok(false);
    }
    let sd = (centered_sum_squares / (sample_count as f64 - 1.0)).sqrt();
    for left in 0..sample_count {
        let left_z = (genotype_column[left] - mean) / sd;
        let row_start = left * sample_count;
        for right in 0..sample_count {
            let right_z = (genotype_column[right] - mean) / sd;
            covariance_sum[row_start + right] += left_z * right_z;
        }
    }
    Ok(true)
}

#[pyfunction]
#[pyo3(signature = (
    sample_nodes,
    num_demes,
    q,
    migration_edges,
    migration_rates,
    num_snps,
    ploidy,
    mutation_rate,
    symmetric,
    seed = None,
    max_attempts = 1000,
    min_allele_frequency = 0.0,
    max_allele_frequency = 1.0,
    marker_model = "first_mutation"
))]
#[allow(clippy::too_many_arguments)]
fn simulate_island_model(
    py: Python<'_>,
    sample_nodes: PyReadonlyArray1<i32>,
    num_demes: usize,
    q: PyReadonlyArray1<f64>,
    migration_edges: PyReadonlyArray2<i32>,
    migration_rates: PyReadonlyArray1<f64>,
    num_snps: usize,
    ploidy: usize,
    mutation_rate: f64,
    symmetric: bool,
    seed: Option<u64>,
    max_attempts: usize,
    min_allele_frequency: f64,
    max_allele_frequency: f64,
    marker_model: &str,
) -> PyResult<(Py<PyArray2<u8>>, Py<PyArray2<u32>>, usize)> {
    if num_demes == 0 {
        return Err(PyValueError::new_err("num_demes must be positive"));
    }
    if num_snps == 0 {
        return Err(PyValueError::new_err("num_snps must be positive"));
    }
    if ploidy == 0 {
        return Err(PyValueError::new_err("ploidy must be positive"));
    }
    if mutation_rate <= 0.0 || !mutation_rate.is_finite() {
        return Err(PyValueError::new_err(
            "mutation_rate must be finite and positive",
        ));
    }
    validate_allele_frequency_filter(min_allele_frequency, max_allele_frequency)?;
    let marker_model = parse_marker_model(marker_model)?;
    let use_random_marker = marker_model != MarkerModel::FirstMutation;
    let condition_on_mutation = marker_model == MarkerModel::ConditionedRandomMarker;

    let sample_nodes = sample_nodes.as_slice()?;
    let q = q.as_slice()?;
    if q.len() != num_demes {
        return Err(PyValueError::new_err("q must have shape (num_demes,)"));
    }
    let edges = migration_edges.as_array();
    let rates = migration_rates.as_slice()?;
    if edges.shape()[1] != 2 || edges.shape()[0] != rates.len() {
        return Err(PyValueError::new_err(
            "migration_edges must have shape (num_edges, 2) and rates must match",
        ));
    }

    let (adjacency, outgoing_totals) = build_adjacency(num_demes, edges, rates, symmetric)?;
    for sample_deme in sample_nodes {
        if *sample_deme < 0 || *sample_deme as usize >= num_demes {
            return Err(PyValueError::new_err(
                "sample_nodes entries must be in [0, num_demes)",
            ));
        }
    }
    for value in q {
        if *value < 0.0 || !value.is_finite() {
            return Err(PyValueError::new_err(
                "q must contain finite non-negative coalescence rates",
            ));
        }
    }

    let mut rng = match seed {
        Some(value) => SmallRng::seed_from_u64(value),
        None => SmallRng::from_entropy(),
    };
    let mut allele_counts = vec![0_u32; num_snps * num_demes];
    let mut accepted = 0_usize;
    let mut attempts = 0_usize;
    let use_bitmask = sample_nodes.len().saturating_mul(ploidy) <= 128;
    while accepted < num_snps && attempts < max_attempts {
        attempts += 1;
        let counts = if use_random_marker && use_bitmask {
            simulate_one_counts_random_marker_bitmask(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
                condition_on_mutation,
            )
        } else if use_random_marker {
            simulate_one_counts_random_marker_words(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
                condition_on_mutation,
            )
        } else if use_bitmask {
            simulate_one_counts_bitmask(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
            )
        } else {
            simulate_one_counts_words(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
            )
        };
        if let Some(counts) = counts {
            if !counts_pass_allele_frequency(
                &counts,
                sample_nodes,
                ploidy,
                min_allele_frequency,
                max_allele_frequency,
            ) {
                continue;
            }
            let start = accepted * num_demes;
            allele_counts[start..start + num_demes].copy_from_slice(&counts);
            accepted += 1;
        }
    }
    if accepted < num_snps {
        return Err(PyRuntimeError::new_err(format!(
            "island simulator accepted {accepted} segregating SNPs after {attempts} attempts; requested {num_snps}",
        )));
    }

    let genotypes = counts_to_genotypes(
        &mut rng,
        &allele_counts,
        num_snps,
        num_demes,
        sample_nodes,
        ploidy,
    )?;
    let genotype_array = Array2::from_shape_vec((sample_nodes.len(), num_snps), genotypes)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let allele_array = Array2::from_shape_vec((num_snps, num_demes), allele_counts)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok((
        genotype_array.into_pyarray(py).to_owned(),
        allele_array.into_pyarray(py).to_owned(),
        attempts,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    sample_nodes,
    num_demes,
    q,
    migration_edges,
    migration_rates,
    num_snps,
    ploidy,
    mutation_rate,
    symmetric,
    seed = None,
    max_attempts = 1000,
    min_allele_frequency = 0.0,
    max_allele_frequency = 1.0,
    marker_model = "first_mutation"
))]
#[allow(clippy::too_many_arguments)]
fn simulate_island_covariance(
    py: Python<'_>,
    sample_nodes: PyReadonlyArray1<i32>,
    num_demes: usize,
    q: PyReadonlyArray1<f64>,
    migration_edges: PyReadonlyArray2<i32>,
    migration_rates: PyReadonlyArray1<f64>,
    num_snps: usize,
    ploidy: usize,
    mutation_rate: f64,
    symmetric: bool,
    seed: Option<u64>,
    max_attempts: usize,
    min_allele_frequency: f64,
    max_allele_frequency: f64,
    marker_model: &str,
) -> PyResult<(Py<PyArray2<f64>>, usize, usize)> {
    if num_demes == 0 {
        return Err(PyValueError::new_err("num_demes must be positive"));
    }
    if num_snps == 0 {
        return Err(PyValueError::new_err("num_snps must be positive"));
    }
    if ploidy == 0 {
        return Err(PyValueError::new_err("ploidy must be positive"));
    }
    if mutation_rate <= 0.0 || !mutation_rate.is_finite() {
        return Err(PyValueError::new_err(
            "mutation_rate must be finite and positive",
        ));
    }
    validate_allele_frequency_filter(min_allele_frequency, max_allele_frequency)?;
    let marker_model = parse_marker_model(marker_model)?;
    let use_random_marker = marker_model != MarkerModel::FirstMutation;
    let condition_on_mutation = marker_model == MarkerModel::ConditionedRandomMarker;

    let sample_nodes = sample_nodes.as_slice()?;
    if sample_nodes.len() < 2 {
        return Err(PyValueError::new_err(
            "sample_nodes must contain at least two samples",
        ));
    }
    let q = q.as_slice()?;
    if q.len() != num_demes {
        return Err(PyValueError::new_err("q must have shape (num_demes,)"));
    }
    for sample_deme in sample_nodes {
        if *sample_deme < 0 || *sample_deme as usize >= num_demes {
            return Err(PyValueError::new_err(
                "sample_nodes entries must be in [0, num_demes)",
            ));
        }
    }
    for value in q {
        if *value < 0.0 || !value.is_finite() {
            return Err(PyValueError::new_err(
                "q must contain finite non-negative coalescence rates",
            ));
        }
    }

    let edges = migration_edges.as_array();
    let rates = migration_rates.as_slice()?;
    let (adjacency, outgoing_totals) = build_adjacency(num_demes, edges, rates, symmetric)?;
    let (samples_by_deme, sampled_demes) = sample_layout(sample_nodes, num_demes);

    let mut rng = match seed {
        Some(value) => SmallRng::seed_from_u64(value),
        None => SmallRng::from_entropy(),
    };
    let use_bitmask = sample_nodes.len().saturating_mul(ploidy) <= 128;
    let mut covariance_sum = vec![0.0_f64; sample_nodes.len() * sample_nodes.len()];
    let mut genotype_column = vec![0.0_f64; sample_nodes.len()];
    let mut effective_snps = 0_usize;
    let mut attempts = 0_usize;

    while effective_snps < num_snps && attempts < max_attempts {
        attempts += 1;
        let counts = if use_random_marker && use_bitmask {
            simulate_one_counts_random_marker_bitmask(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
                condition_on_mutation,
            )
        } else if use_random_marker {
            simulate_one_counts_random_marker_words(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
                condition_on_mutation,
            )
        } else if use_bitmask {
            simulate_one_counts_bitmask(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
            )
        } else {
            simulate_one_counts_words(
                &mut rng,
                sample_nodes,
                num_demes,
                q,
                &adjacency,
                &outgoing_totals,
                ploidy,
                mutation_rate,
            )
        };
        if let Some(counts) = counts {
            if !counts_pass_allele_frequency(
                &counts,
                sample_nodes,
                ploidy,
                min_allele_frequency,
                max_allele_frequency,
            ) {
                continue;
            }
            let usable = accumulate_covariance_from_counts(
                &mut rng,
                &counts,
                &samples_by_deme,
                &sampled_demes,
                ploidy,
                &mut genotype_column,
                &mut covariance_sum,
            )?;
            if usable {
                effective_snps += 1;
            }
        }
    }
    if effective_snps < num_snps {
        return Err(PyRuntimeError::new_err(format!(
            "island covariance simulator accepted {effective_snps} usable SNPs after {attempts} attempts; requested {num_snps}",
        )));
    }
    for value in &mut covariance_sum {
        *value /= effective_snps as f64;
    }
    let covariance_array =
        Array2::from_shape_vec((sample_nodes.len(), sample_nodes.len()), covariance_sum)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok((
        covariance_array.into_pyarray(py).to_owned(),
        effective_snps,
        attempts,
    ))
}

#[pymodule]
fn _island_sim(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(simulate_island_model, module)?)?;
    module.add_function(wrap_pyfunction!(simulate_island_covariance, module)?)?;
    Ok(())
}
