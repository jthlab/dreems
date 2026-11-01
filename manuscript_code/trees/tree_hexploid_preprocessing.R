# This code is borrowed from Cummins, et al. (2025) - Patterns of genotype-environment 
# association in the eastern North American yellow birch (Betula alleghaniensis Britt.)
# For proprocessing the data, we followed near exactly how the original authors suggested

library(polyRAD)
library(VariantAnnotation)
library(pcaMethods)

# this was where I stored the downloaded data
myVCF <- "/Users/jkliang/Downloads/second_filters_m4_p80_x0_S3.singleton.unlinked_100k_0.5.imputed_k1.vcf"

# The only change we made to the data was removing the possibility of having diploids
mydata <- VCF2RADdata(myVCF, 
                      phaseSNPs = FALSE,
                      # possiblePloidies = list(2,6,c(2,2,2)),
                      possiblePloidies = list(6,c(2,2,2)),
                      expectedLoci = 40000,
                      expectedAlleles = 200000,
                      min.ind.with.reads = 170,
                      min.ind.with.minor.allele = 3,
                      taxaPloidy = 2)
# Overdispersion testing
overdispersionP <- TestOverdispersion(mydata, to_test = 3:12)
sapply(overdispersionP[names(overdispersionP) != "optimal"], quantile, probs = c(0.01, 0.25, 0.5, 0.75, 0.99))
my_ovdisp <- overdispersionP$optimal

# Hind/He filtering
myhindhe <- HindHe(mydata)
myhindheByLoc <- colMeans(myhindhe, na.rm = TRUE)
new_cutoff <- 0.833  #cutoff threshold (ploidy-1/ploidy)
valid_loci_new <- myhindheByLoc <= new_cutoff
loci_to_keep_new <- which(valid_loci_new)
mydata_cutoff_filtered <- SubsetByLocus(mydata, loci = loci_to_keep_new)

# Re-test overdispersion
overdispersionP2 <- TestOverdispersion(mydata_cutoff_filtered, to_test = 3:12)
sapply(overdispersionP2[names(overdispersionP2) != "optimal"], quantile, probs = c(0.01, 0.25, 0.5, 0.75, 0.99))
my_ovdisp2 <- overdispersionP2$optimal

# Remove hybrid taxa
all_taxa <- GetTaxa(mydata)
taxa_to_remove <- c("YB_113-17", "YB_113-2", "YB_116-4")
taxa_to_keep <- all_taxa[!all_taxa %in% taxa_to_remove]
mydata2 <- SubsetByTaxon(mydata_cutoff_filtered, taxa_to_keep)

# PCA on population structure
mydataPopStruct <- IteratePopStruct(mydata2, nPcsInit = 8, tol = 5e-03, overdispersion = my_ovdisp2)

# where I stored the filtered hexaploid data
RADdata2VCF(
  mydataPopStruct,
  file = "/Users/jkliang/Downloads/filtered_tree_genotypes.vcf"
)
