################################################################################
# AUTOSYNTH SIMULATION FRAMEWORK - JULY 2024
# Purpose: Synthetic data generation and Performance Benchmarking (Stress Test)
################################################################################

library(MASS)
library(Matrix)
library(keras)
library(tensorflow)

# ---- 1. SETUP PARAMETERS ----

k <- 10
u <- 1000
numerosity <- c(50, 250, 1000) # As requested

# ---- 2. SYNTHETIC DATA GENERATION ----

# --- D1: Independent and Identically Distributed (IID) ---
data_iid <- list()
for(jj in 1:100){
  set.seed(jj)
  x_mat <- matrix(nrow=u, ncol=k)
  for (ii in 1:k){
    x_mat[,ii] <- rnorm(u, mean=runif(1, -100, 100), sd=runif(1, 1, 100))  
  }
  data_iid[[jj]] <- x_mat
}
save(data_iid, file="iid.RData")

# --- D2: Dependent ID (Random Correlation) ---
data_2 <- list()
for(jj in 1:100){
  set.seed(jj)
  mu_local <- runif(k , -100, 100)
  sigma_mat <- matrix(runif(k^2, -1, 1), nrow=k)
  ss <- nearPD(sigma_mat) # Ensure positive definiteness
  sigma_mat <- as.matrix(ss$mat)
  data_2[[jj]] <- mvrnorm(n = u, mu = mu_local, Sigma = sigma_mat)
}
save(data_2, file="id_rnd_correlation.RData")

# --- D3: Dependent ID (Gaussian Process Kernel) ---
data_3 <- list()
gaussian_kernel <- function(dmat, range_param, scale_param) {
  return(scale_param * exp(-abs(dmat) / (2 * range_param ^ 2)))
}

for(jj in 1:100){
  set.seed(jj)
  points <- sort(runif(k, -10, 10) / 10)
  xd <- as.matrix(dist(points))
  cov_matrix_gp <- gaussian_kernel(xd, .55, 2)
  mu_gp <- runif(k, -100, 100)
  data_3[[jj]] <- mvrnorm(n=u, mu=mu_gp, Sigma=cov_matrix_gp)
}
save(data_3, file="id_kernel.RData")

# --- D4: Non-IID (Mixed Distributions) ---
data_4 <- list()
for(jj in 1:100){
  set.seed(jj)
  n_max <- numerosity[3]
  
  # Generation of different distributions
  norm_part  <- matrix(rnorm(n_max * 8, mean=runif(8, -1000, 1000), sd=runif(8, 0.1, 100)), ncol=8)
  unif_part  <- runif(n_max, min = 0, max = 1)
  binom_part <- matrix(runif(n_max * 5, min = 0, max = 1), ncol=5)
  pois_part  <- matrix(rpois(n_max * 3, lambda = runif(3, 1, 50)), ncol=3)
  exp_part   <- matrix(rexp(n_max * 3, rate = runif(3, 1, 50)), ncol=3)
  chi_part   <- matrix(rchisq(n_max * 3, df = runif(3, 1, 50)), ncol=3)
  geom_part  <- matrix(rgeom(n_max * 3, prob = runif(3, 0, 1)), ncol=3)
  
  xx <- cbind(norm_part, unif_part, binom_part, pois_part, exp_part, chi_part, geom_part)
  
  set.seed(0)
  choice <- sample(seq(1, ncol(xx), 1), k)
  data_4[[jj]] <- xx[, choice]
}
save(data_4, file = "non_iid.RData")

# ---- 3. SIMULATION FUNCTION ----

simulations_july <- function(data_list, B, norm_ae = TRUE) {
  stress_total_dist <- list()
  stress_total_rank <- list()
  
  # Loop over the 3 different sample sizes defined in numerosity
  for (nn in 1:3) {
    current_n <- numerosity[nn]
    
    d_stress_results <- matrix(ncol=4, nrow=length(data_list))
    r_stress_results <- matrix(ncol=4, nrow=length(data_list))
    
    for (jj in 1:length(data_list)) {
      # Subsampling current dataset
      sx <- sample(1:nrow(data_list[[jj]]), current_n)
      loc_data <- data_list[[jj]][sx, ]
      
      iter_dist_vals <- vector()
      iter_rank_vals <- vector()
      
      for (ii in 1:B) {
        # Autoencoder Setup
        ae_iters <- ifelse(norm_ae, 1, 30)
        enc <- autoencoder(ncol(loc_data), 1, loc_data, ae_iters, 1, 32, 0.2, "linear")
        
        # Predictions
        as_pred   <- predict(enc$autoencoder, loc_data)
        pca_pred  <- pca_weight(loc_data)
        ampi_pred <- ampi(norm_ampi(loc_data))
        ar_pred   <- apply(norm_ampi(loc_data), 1, mean)
        
        # Distance Stress Calculation
        res_dist <- c(
          stress(loc_data, as_pred),
          stress(loc_data, pca_pred),
          stress(norm_ampi(loc_data), ampi_pred),
          stress(norm_ampi(loc_data), ar_pred)
        )
        iter_dist_vals <- rbind(iter_dist_vals, res_dist)
        
        # Rank Stress Calculation (Calling the rank_stress function)
        res_rank <- c(
          rank_stress(loc_data, as_pred),
          rank_stress(loc_data, pca_pred),
          rank_stress(norm_ampi(loc_data), ampi_pred),
          rank_stress(norm_ampi(loc_data), ar_pred)
        )
        iter_rank_vals <- rbind(iter_rank_vals, res_rank)
      }
      
      # Compute median performance across B iterations
      d_stress_results[jj, ] <- apply(iter_dist_vals, 2, median)
      r_stress_results[jj, ] <- apply(iter_rank_vals, 2, median)
    }
    
    stress_total_dist[[nn]] <- d_stress_results
    stress_total_rank[[nn]] <- r_stress_results
    message(paste("Iteration for N =", current_n, "completed."))
  }
  
  return(list(dist = stress_total_dist, rank = stress_total_rank))
}

# ---- 4. EXECUTION ----

# Case 1: IID
prova <- simulations_july(data_iid, B=100, norm_ae=FALSE)

# Case 2: ID Kernel
sim_id_kernel <- simulations_july(data_3, B=100, norm_ae=FALSE)

# Case 3: Non-IID
sim_nonid <- simulations_july(data_4, B=100, norm_ae=FALSE)

# Quick visualization of the first scenario
hist(sim_id_kernel$dist[[1]][, 1], xlim=c(0, 1), main="Autoencoder Stress (N=50)", col="skyblue")