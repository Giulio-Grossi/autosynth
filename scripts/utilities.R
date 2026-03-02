# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality Reduction & Index Construction Utilities
# ─────────────────────────────────────────────────────────────────────────────

library(psych)
library(dplyr)


# ------------------------------------------------------------------------------
# pca_linear
# Performs a simple PCA on scaled data and returns the projection of the data
# onto the first principal component loadings.
# ------------------------------------------------------------------------------
pca_linear <- function(data) {
  df_scaled <- scale(data)
  pca <- principal(df_scaled, rotate = "none")
  out <- df_scaled %*% pca$loading
  return(out)
}


# ------------------------------------------------------------------------------
# pca_weight
# Performs PCA and constructs a single composite index by weighting the
# principal components (explaining >= 90% of variance) by their relative
# standard deviations.
# ------------------------------------------------------------------------------
pca_weight <- function(X) {
  X_scaled <- scale(X)
  
  # Perform PCA with centering and scaling
  pca_result <- prcomp(X, center = TRUE, scale. = TRUE)
  
  # Compute cumulative proportion of variance explained
  cumulative_variance <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
  
  # Select the minimum number of components explaining >= 90% of variance
  threshold_variance <- 0.90
  num_components <- which(cumulative_variance >= threshold_variance)[1]
  
  # Project data onto the selected components
  y <- X %*% pca_result$rotation[, 1:num_components]
  
  # Weight each component by its relative standard deviation
  weights <- pca_result$sdev[1:num_components] / sum(pca_result$sdev[1:num_components])
  out <- y %*% weights
  
  return(out)
}


# ------------------------------------------------------------------------------
# ampi
# Computes the AMPI (Adjusted Mean-based Performance Index):
# for each observation, combines the row mean and row standard deviation
# adjusted by the coefficient of variation (CV = sd / mean).
# ------------------------------------------------------------------------------
ampi <- function(dt) {
  mm  <- apply(dt, 1, mean)
  sd  <- apply(dt, 1, sd)
  cv1 <- sd / mm
  out <- mm + sd * cv1
  return(out)
}


# ------------------------------------------------------------------------------
# stress
# Computes the Kruskal stress measure between the original multivariate data
# and a compressed 1-dimensional index, using Euclidean distances.
# A lower stress value indicates better preservation of the distance structure.
# ------------------------------------------------------------------------------
stress <- function(data, compressed) {
  oo_dist <- dist(data,       method = "euclidean")
  cc_dist <- dist(compressed, method = "euclidean")
  out <- round((sum((oo_dist - cc_dist)^2) / sum(oo_dist^2)), 4)
  return(out)
}


# ------------------------------------------------------------------------------
# rank_stress
# Same as stress(), but operates on ranked data and ranked index values.
# Useful for a rank-based (non-parametric) assessment of structure preservation.
# ------------------------------------------------------------------------------
rank_stress <- function(data, index) {
  r_data  <- apply(data, 2, rank)
  r_index <- rank(index)
  
  oo_dist <- dist(r_data,   method = "euclidean")
  cc_dist <- dist(r_index,  method = "euclidean")
  out <- round((sum((oo_dist - cc_dist)^2) / sum(oo_dist^2)), 4)
  return(out)
}


# ------------------------------------------------------------------------------
# get_variable_importance
# Extracts variable importance from a Keras autoencoder model by summing
# the absolute values of the weights in the first encoding layer.
# ------------------------------------------------------------------------------
get_variable_importance <- function(autoencoder_model) {
  weights             <- get_weights(autoencoder_model$layers[[2]])[[1]]
  variable_importance <- rowSums(abs(weights))
  return(variable_importance)
}


# ------------------------------------------------------------------------------
# sorter
# Aligns the sign of each column in 'enc' so that the observation with the
# lowest rank in 'aa' also has a low rank in each encoded dimension.
# Ensures consistent directional interpretation across components.
# ------------------------------------------------------------------------------
sorter <- function(enc, aa) {
  rr <- rank(aa)
  for (i in 1:ncol(enc)) {
    loc_r <- rank(enc[, i])
    if (loc_r[which(rr == min(rr))] > (nrow(enc) / 2)) {
      enc[, i] <- -enc[, i]
    }
  }
  return(enc)
}


# ------------------------------------------------------------------------------
# calculate_CFI
# Computes a Comparative Fit Index (CFI)-like statistic comparing the observed
# covariance matrix with a model-implied covariance matrix reconstructed from
# a compressed vector.
# Note: this is an approximation; the estimated covariance is set equal to the
# observed covariance as a baseline.
# ------------------------------------------------------------------------------
calculate_CFI <- function(original_data, compressed_vector) {
  observed_cov_matrix <- cov(original_data)
  num_variables       <- nrow(original_data)
  
  # Reconstruct the model-implied covariance matrix from the compressed vector
  model_implied_cov_matrix <- matrix(NA, nrow = num_variables, ncol = num_variables)
  model_implied_cov_matrix[lower.tri(model_implied_cov_matrix)] <- compressed_vector
  model_implied_cov_matrix <- t(model_implied_cov_matrix)
  model_implied_cov_matrix[lower.tri(model_implied_cov_matrix)] <- compressed_vector
  
  # Use observed covariance as the baseline estimated covariance
  estimated_cov_matrix <- cov(original_data)
  
  n <- nrow(original_data)
  p <- length(compressed_vector)
  
  # Chi-squared statistic
  chi_squared <- sum((observed_cov_matrix - model_implied_cov_matrix)^2 / estimated_cov_matrix)
  
  # Degrees of freedom
  df <- (n * (n + 1) / 2) - p
  
  # CFI approximation
  CFI <- 1 - (chi_squared / df) * ((n - 1) / n)
  return(CFI)
}


# ------------------------------------------------------------------------------
# norm_ampi
# Normalizes each column of the input matrix to the range [70, 130],
# rescaling from [0, 1] after min-max normalization.
# Useful for expressing indices on a standardized bounded scale.
# ------------------------------------------------------------------------------
norm_ampi <- function(rr) {
  for (i in 1:ncol(rr)) {
    rr[, i] <- (rr[, i] - min(rr[, i])) / (max(rr[, i]) - min(rr[, i]))
    rr[, i] <- (rr[, i] * 60) + 70
  }
  return(rr)
}


# ------------------------------------------------------------------------------
# inverti_se_necessario
# Checks whether the maximum values of two index columns fall in the same
# quartile. If not, the sign of the second column is flipped to ensure
# directional consistency between the two indices.
# ------------------------------------------------------------------------------
inverti_se_necessario <- function(matrice) {
  indice1 <- matrice[, 1]
  indice2 <- matrice[, 2]
  
  # Assign quartile labels to each observation
  quartili_indice1 <- cut(indice1, breaks = quantile(indice1, probs = 0:4/4),
                          include.lowest = TRUE, labels = FALSE)
  quartili_indice2 <- cut(indice2, breaks = quantile(indice2, probs = 0:4/4),
                          include.lowest = TRUE, labels = FALSE)
  
  # Find the quartile of the maximum value in each column
  quartile_max_indice1 <- quartili_indice1[which.max(indice1)]
  quartile_max_indice2 <- quartili_indice2[which.max(indice2)]
  
  # If maxima fall in different quartiles, flip the sign of the second index
  if (quartile_max_indice1 != quartile_max_indice2) {
    matrice[, 2] <- -matrice[, 2]
    message("Alternative index sign flipped: maxima were in different quartiles.")
  } else {
    message("Maxima are in the same quartile; no modification applied.")
  }
  
  return(matrice)
}
