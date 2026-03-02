# ─────────────────────────────────────────────────────────────────────────────
# Index Construction via Autoencoder: Stress Test over Concurrent Methods
#
# DESCRIPTION:
#   This script runs a Monte Carlo stress test (500 iterations) comparing three
#   dimensionality-reduction methods for composite index construction:
#     1. Autoencoder (neural network-based)
#     2. PCA-weighted index (pca_weight)
#     3. AMPI index (ampi)
#   The comparison is performed on both raw and normalized input data.
#   Stress values (Kruskal) are used to assess how well each method preserves
#   the multivariate distance structure of the original data.
#
# INPUTS:
#   - dataset_usa.R         : script that prepares the raw and normalized datasets
#   - normed_data_usa.RData : normalized dataset (ampi_data)
#   - raw_data_usa.RData    : raw dataset (raw_data)
#   - autoencoder_new.R     : autoencoder() function definition
#   - utilities.R           : helper functions (stress, ampi, norm_ampi,
#                             pca_weight, get_variable_importance,
#                             inverti_se_necessario)
#
# OUTPUTS:
#   - raw_usa500_feb25.RData    : stress test results on raw data (500 runs)
#   - normed_usa500_feb25.RData : stress test results on normalized data (500 runs)
#   Each saved object (final) contains:
#     $store               : list of per-iteration index matrices [enc, pca, ampi]
#     $sorted_store        : list of sign-aligned index matrices
#     $input_weights       : variable importance weights from the encoder
#     $reconstruction_error: per-variable reconstruction loss
#     $stress_total        : stress values for all four methods across iterations
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────────────────────────────────────

library(reticulate)
library(keras3)
library(tensorflow)

source("scripts/dataset_usa.R")
source("scripts/autoencoder.R")
source("scripts/utilities.R")

load("datasets/normed_data_usa.RData")
load("datasets/raw_data_usa.RData")

# Strip identifier columns (assumed to be columns 1 and 2)
raw    <- as.matrix(raw_data[, -c(1:2)])
normed <- ampi_data[, -c(1:2)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. STRESS TEST ON RAW DATA (500 iterations)
# ─────────────────────────────────────────────────────────────────────────────

bb                  <- 500
store               <- list()
stress_total        <- vector()
weights             <- vector()
reconstruction_error <- vector()

for (ii in 1:bb) {
  kk <- as.matrix(raw)
  
  # Train autoencoder (1 bottleneck unit, 5 layers, 32 hidden units, 0.2 dropout)
  enc <- autoencoder(ncol(kk), 1, kk, iter = 15, layers = 5, 32, 0.2, "linear")
  
  # Encode input data to the bottleneck representation
  loc <- enc$encoder %>% predict(kk)
  
  # Reconstruct input and compute per-variable mean reconstruction error
  x_hat <- enc$autoencoder %>% predict(kk)
  loss  <- apply(x_hat - kk, 2, mean)
  
  # Extract variable importance from encoder weights
  loc_weights <- get_variable_importance(enc$encoder)
  
  # Compute PCA-weighted index and AMPI index
  pp <- pca_weight(kk)
  aa <- ampi(norm_ampi(kk))
  
  # Compute Kruskal stress for all four methods
  res1a <- c(
    stress(kk,           loc),
    stress(kk,           pp),
    stress(norm_ampi(kk), aa),
    stress(norm_ampi(kk), rowMeans(norm_ampi(kk)))
  )
  
  # Accumulate results
  stress_total         <- rbind(stress_total, res1a)
  weights              <- cbind(weights, loc_weights)
  reconstruction_error <- cbind(reconstruction_error, loss)
  store[[ii]]          <- cbind(loc, pp, aa)
  
  print(paste0("Progress: ", round(ii / bb * 100, 2), " %"))
}

# ── Sign alignment across iterations ─────────────────────────────────────────
# Ensure that the autoencoder and PCA indices are directionally consistent
# with the AMPI index (used as reference) in each iteration.

sorted_store <- list()

for (ii in 1:bb) {
  loc <- store[[ii]]
  
  # Pair each index with AMPI as reference (column 3)
  matrice_enc <- cbind(loc[, 3], loc[, 1])
  matrice_pca <- cbind(loc[, 3], loc[, 2])
  
  sorted_enc <- inverti_se_necessario(matrice_enc)[, 2]
  sorted_pca <- inverti_se_necessario(matrice_pca)[, 2]
  aa         <- store[[ii]][, 3]
  
  sorted_store[[ii]] <- cbind(sorted_enc, sorted_pca, aa)
}

# ── Save results ──────────────────────────────────────────────────────────────

final                      <- list()
final$store                <- store
final$sorted_store         <- sorted_store
final$input_weights        <- weights
final$reconstruction_error <- reconstruction_error
final$stress_total         <- stress_total

save(final, file = "results/raw_usa500.RData")


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRESS TEST ON NORMALIZED DATA (500 iterations)
# ─────────────────────────────────────────────────────────────────────────────

bb                   <- 500
store                <- list()
stress_total         <- vector()
weights              <- vector()
reconstruction_error <- vector()

for (ii in 1:bb) {
  kk <- as.matrix(normed)
  
  # Train autoencoder
  enc <- autoencoder(ncol(kk), 1, kk, iter = 15, layers = 5, 32, 0.2, "linear")
  
  # Extract encoder weights and bottleneck representation
  loc_weights <- get_variable_importance(enc$encoder)
  loc         <- enc$encoder %>% predict(kk)
  
  # Reconstruct input and compute per-variable mean reconstruction error
  x_hat <- enc$autoencoder %>% predict(kk)
  loss  <- apply(x_hat - kk, 2, mean)
  
  # Compute PCA-weighted index and AMPI index
  pp <- pca_weight(kk)
  aa <- ampi(norm_ampi(kk))
  
  # Compute Kruskal stress for all four methods
  res1a <- c(
    stress(kk,            loc),
    stress(kk,            pp),
    stress(norm_ampi(kk), aa),
    stress(norm_ampi(kk), rowMeans(norm_ampi(kk)))
  )
  
  # Accumulate results
  stress_total         <- rbind(stress_total, res1a)
  weights              <- cbind(weights, loc_weights)
  reconstruction_error <- cbind(reconstruction_error, loss)
  store[[ii]]          <- cbind(loc, pp, aa)
  
  print(paste0("Progress: ", round(ii / bb * 100, 2), " %"))
}

# ── Sign alignment across iterations ─────────────────────────────────────────

sorted_store <- list()

for (ii in 1:bb) {
  loc <- store[[ii]]
  
  matrice_enc <- cbind(loc[, 3], loc[, 1])
  matrice_pca <- cbind(loc[, 3], loc[, 2])
  
  sorted_enc <- inverti_se_necessario(matrice_enc)[, 2]
  sorted_pca <- inverti_se_necessario(matrice_pca)[, 2]
  aa         <- store[[ii]][, 3]
  
  sorted_store[[ii]] <- cbind(sorted_enc, sorted_pca, aa)
}

# ── Save results ──────────────────────────────────────────────────────────────

final                      <- list()
final$store                <- store
final$sorted_store         <- sorted_store
final$input_weights        <- weights
final$reconstruction_error <- reconstruction_error
final$stress_total         <- stress_total

save(final, file = "results/normed_usa500.RData")


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST-HOC DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

# ── 3A. Reconstruction error weights (normalized per iteration) ───────────────
# Each variable's reconstruction error is expressed as a share of the total
# error in that iteration, to assess relative variable-level fit.

o_weights <- abs(final$reconstruction_error)
sum_w     <- apply(o_weights, 2, sum)

for (ii in 1:ncol(o_weights)) {
  o_weights[, ii] <- o_weights[, ii] / sum_w[ii]
}

# Plot distribution of normalized reconstruction error for each variable
par(mfrow = c(4, 3))
par(mar  = c(1, 1, 1, 1))

for (ii in 1:nrow(o_weights)) {
  hist(o_weights[ii, ], col = "aquamarine2", main = "", ylab = "")
}

# Summary quantiles across iterations
round(apply(o_weights, 1, quantile), 2)


# ── 3B. Input variable importance weights (normalized per iteration) ──────────
# Each variable's importance is expressed as a share of the total importance
# in that iteration, derived from the encoder's first-layer weights.

i_weights <- final$input_weights
sum_w     <- apply(i_weights, 2, sum)

for (ii in 1:ncol(i_weights)) {
  i_weights[, ii] <- i_weights[, ii] / sum_w[ii]
}

# Plot distribution of normalized input weights for each variable
par(mfrow = c(4, 3))
par(mar  = c(1, 1, 1, 1))

for (ii in 1:nrow(i_weights)) {
  hist(i_weights[ii, ], col = "aquamarine2", main = "", ylab = "")
}

apply(i_weights, 1, quantile)


# ── 3C. Stress distribution comparison across methods ────────────────────────
# Histogram of autoencoder stress values, with vertical lines showing the
# mean stress of PCA (darkred), AMPI (royalblue), and row-mean (springgreen3).

hist(stress_total[, 1], xlim = c(0.5, 1), col = "goldenrod", breaks = 20)
abline(v = mean(stress_total[, 2]), col = "darkred",     lwd = 2)  # PCA
abline(v = mean(stress_total[, 3]), col = "royalblue",   lwd = 2)  # AMPI
abline(v = mean(stress_total[, 4]), col = "springgreen3", lwd = 2) # Row mean

