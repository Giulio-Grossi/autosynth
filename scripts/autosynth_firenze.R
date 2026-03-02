################################################################################
# AUTOSYNTH INDEX CALCULATION - FLORENCE CASE STUDY
# Purpose: Stress testing and weight analysis using Autoencoders, PCA, and AMPI
################################################################################

# ---- 1. ENVIRONMENT SETUP & DATA IMPORT ----

library(reticulate)
library(keras)
library(tensorflow)

# Load custom functions
source("autoencoder_new.R")
source("utilities.R")

# Load pre-processed data for Florence
load("dati_firenze_oct24.RData")
raw    <- dati_firenze$raw
normed <- dati_firenze$normed

# Define variables of interest for the index
vars <- c("reddito_ind_median", "reddito_fam_mediano", "anziani",
          "delta residenti", "saldo naturale", "reddito basso", "affitto",
          "anziani soli", "giovani stranieri", "monogenitore",
          "sfitti", "laureati", "stanzialita")

# Subset data for analysis
raw_autoenc    <- raw[, vars]
# (Removed initial normalization here as per instructions)

# ---- 2. STRESS TEST: RAW DATA (1-A) ----

bb <- 500  # Number of iterations
store <- list()
stress_total <- vector()
weights <- vector()
reconstruction_error <- vector()

for(ii in 1:bb) {
  kk <- as.matrix(raw_autoenc)
  
  # Train Autoencoder (Latent dimension = 1 for index compression)
  enc <- autoencoder(ncol(kk), 1, kk, iter=35, layers=100, batch=32, 
                     validation=0.2, activation="linear")
  
  # Predictions and variable importance
  loc         <- enc$encoder %>% predict(kk)
  x_hat       <- enc$autoencoder %>% predict(kk)
  loss        <- apply(x_hat - kk, 2, mean)
  loc_weights <- get_variable_importance(enc$encoder)
  
  # Alternative Methods comparison
  pp <- pca_weight(kk)                        
  aa <- ampi(norm_ampi(kk))                   
  
  # Calculate Stress Metrics
  res1a <- c(stress(kk, loc),                                
             stress(kk, pp),                                 
             stress(norm_ampi(kk), aa),                      
             stress(norm_ampi(kk), rowMeans(norm_ampi(kk)))) 
  
  # Store iteration results
  stress_total         <- rbind(stress_total, res1a)
  weights              <- cbind(weights, loc_weights)
  reconstruction_error <- cbind(reconstruction_error, loss)
  store[[ii]]          <- cbind(loc, pp, aa)
  
  print(paste0("Progress (Raw Data): ", round(ii/bb*100, 2), " %"))
}

# --- Polarity Alignment for Raw Data ---
sorted_store <- list()
for(ii in 1:bb) {
  loc         <- store[[ii]]
  matrice_enc <- cbind(loc[, 3], loc[, 1])
  matrice_pca <- cbind(loc[, 3], loc[, 2])
  
  # Invert orientation if necessary to match AMPI reference
  sorted_enc <- (inverti_se_necessario(matrice_enc)[, 2])
  sorted_pca <- (inverti_se_necessario(matrice_pca)[, 2])
  aa         <- store[[ii]][, 3]
  sorted_store[[ii]] <- cbind(sorted_enc, sorted_pca, aa)
}

# Export Raw Analysis Results
final_raw <- list(
  store                = store,
  sorted_store         = sorted_store,
  input_weights        = weights,
  reconstruction_error = reconstruction_error,
  stress_total         = stress_total
)
save(final_raw, file="raw_fi500_feb25.RData")

# ---- 3. STRESS TEST: NORMED DATA ----

# Reset storage for the second simulation (Normalized Data)
store <- list()
stress_total <- vector()
weights <- vector()
reconstruction_error <- vector()

# Apply Min-Max normalization
normed_autoenc_proc <- norm_ampi(normed[, vars])

for(ii in 1:bb) {
  kk <- as.matrix(normed_autoenc_proc)
  
  # Train Autoencoder on normalized data
  enc <- autoencoder(ncol(kk), 1, kk, iter=35, layers=100, batch=32, 
                     validation=0.2, activation="linear")
  
  loc_weights <- get_variable_importance(enc$encoder)
  loc         <- enc$encoder %>% predict(kk)
  x_hat       <- enc$autoencoder %>% predict(kk)
  loss        <- apply(x_hat - kk, 2, mean)
  
  # Alternative Methods comparison
  pp <- pca_weight(kk)
  aa <- ampi(norm_ampi(kk))
  
  # Stress metrics calculation
  res1a <- c(stress(kk, loc),
             stress(kk, pp),
             stress(norm_ampi(kk), aa),
             stress(norm_ampi(kk), rowMeans(norm_ampi(kk))))
  
  stress_total         <- rbind(stress_total, res1a)
  weights              <- cbind(weights, loc_weights)
  reconstruction_error <- cbind(reconstruction_error, loss)
  store[[ii]]          <- cbind(loc, pp, aa)
  
  print(paste0("Progress (Normed Data): ", round(ii/bb*100, 2), " %"))
}

# --- Polarity Alignment for Normed Data ---
sorted_store <- list()
for(ii in 1:bb) {
  loc         <- store[[ii]]
  matrice_enc <- cbind(loc[, 3], loc[, 1])
  matrice_pca <- cbind(loc[, 3], loc[, 2])
  
  sorted_enc <- (inverti_se_necessario(matrice_enc)[, 2])
  sorted_pca <- (inverti_se_necessario(matrice_pca)[, 2])
  aa         <- store[[ii]][, 3]
  sorted_store[[ii]] <- cbind(sorted_enc, sorted_pca, aa)
}

# Export Normed Analysis Results
final_normed <- list(
  store                = store,
  sorted_store         = sorted_store,
  input_weights        = weights,
  reconstruction_error = reconstruction_error,
  stress_total         = stress_total
)
save(final_normed, file="normed_fi500_feb25.RData")

# ---- 4. FINAL WEIGHTS ANALYSIS (Using final_normed) ----

# Reconstruction error distribution analysis
o_weights <- abs(final_normed$reconstruction_error)
sum_w     <- apply(o_weights, 2, sum)

for(ii in 1:ncol(o_weights)) {
  o_weights[, ii] = o_weights[, ii] / sum_w[ii]
}

# Plot Histograms for Reconstruction Weights
par(mfrow=c(4, 3), mar=c(1, 1, 1, 1))
for(ii in 1:nrow(o_weights)) {
  hist(o_weights[ii, ], col="aquamarine2", main="", ylab="")
}

# Input Weights (Variable Importance) Analysis
i_weights <- final_normed$input_weights
sum_w     <- apply(i_weights, 2, sum)

for(ii in 1:ncol(i_weights)) {
  i_weights[, ii] = i_weights[, ii] / sum_w[ii]
}

# Plot Histograms for Input Weights
par(mfrow=c(4, 3), mar=c(1, 1, 1, 1))
for(ii in 1:nrow(i_weights)) {
  hist(i_weights[ii, ], col="aquamarine2", main="", ylab="")
}

# Overall Stress Comparison Plot
hist(stress_total[, 1], xlim=c(0.5, 1), col="goldenrod", breaks=20, 
     main="Stress Comparison (Normed)", xlab="Stress Value")
abline(v=mean(stress_total[, 2]), col="darkred", lwd=2)      # PCA Mean
abline(v=mean(stress_total[, 3]), col="royalblue", lwd=2)    # AMPI Mean
abline(v=mean(stress_total[, 4]), col="springgreen3", lwd=2) # Simple Mean Mean