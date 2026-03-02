################################################################################
# VISUALIZATION: SIMULATION RESULTS (JULY 2024)
# Purpose: Generate faceted boxplots for Stress metrics and Sensitivity analysis.
################################################################################

# ---- 1. PLOTTING FUNCTION ----

#' Generate Faceted Boxplots for Methods Comparison
#' 
#' @param data A nested list containing simulation results (dist/rank).
#' @param mode Integer (1 or 2) to select the metric (e.g., Distance vs Rank).
#' @param up Upper limit for the Y-axis.
#' @param down Lower limit for the Y-axis.
#' @param numerosity Vector of sample sizes used in simulations.
#' 
#' @return A ggplot object with faceted boxplots by sample size.
plots_luglio <- function(data, mode, up, down, numerosity) {
  library(ggplot2)
  library(tidyr)
  library(dplyr) # Required for the pipe operator and data manipulation
  
  df_combined <- data.frame()
  
  # Iterate through available sample size results in data[[mode]]
  for (ii in seq_along(data[[mode]])) {
    df <- as.data.frame(data[[mode]][[ii]])
    colnames(df) <- c("Autosynth", "PCA", "AMPI", "Mean")
    
    # Transform to long format for ggplot2 compatibility
    df_long <- df %>%
      pivot_longer(
        cols      = everything(),
        names_to  = "Method",
        values_to = "Value"
      )
    
    # Add metadata for faceting
    df_long$Method <- as.factor(df_long$Method)
    df_long$N      <- paste0("N = ", numerosity[[ii]])
    
    df_combined <- rbind(df_combined, df_long)
  }
  
  # Create combined plot
  g <- ggplot(df_combined, aes(x = Method, y = Value, fill = Method)) +
    geom_boxplot() +
    # Zooming in without removing data points (keeps outliers in calculation)
    coord_cartesian(ylim = c(down, up)) + 
    scale_fill_manual(values = c("#CC79A7", "#56B4E9", "#009E73", "#E69F00")) +
    labs(x = "Method", y = expression(theta)) +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 9),
      axis.title.y = element_text(size = 9),
      strip.text   = element_text(size = 10),
      legend.position = "bottom"
    ) +
    facet_wrap(~ N, ncol = 3)
  
  return(g)
}

# ---- 2. GENERATING SIMULATION PLOTS ----

# Assumes 'ext', 'sim_id_kernel', 'prova' and 'numerosity' are loaded in the environment
# Distance Stress Plots (Mode 1) and Rank Stress Plots (Mode 2)

# Case 3: Mixed Distributions (Non-IID)
stress_3 <- plots_luglio(ext, mode = 1, up = 0.95, down = 0.15, numerosity)
rank_3   <- plots_luglio(ext, mode = 2, up = 0.60, down = 0.50, numerosity)

# Case 2: ID Kernel (Gaussian Process)
stress_2 <- plots_luglio(sim_id_kernel, mode = 1, up = 0.95, down = 0.15, numerosity)
rank_2   <- plots_luglio(sim_id_kernel, mode = 2, up = 0.60, down = 0.50, numerosity)

# Case 1: IID Data
stress_1 <- plots_luglio(prova, mode = 1, up = 0.95, down = 0.15, numerosity)
rank_1   <- plots_luglio(prova, mode = 2, up = 0.60, down = 0.50, numerosity)

# ---- 3. SENSITIVITY ANALYSIS VISUALIZATION ----

# Assumes 'sens_data_4' contains sensitivity matrices
sens <- sens_data_4
g_sens_list <- list()

for(ii in 1:3) {
  sens_nn <- sens[[ii]]
  
  # Calculate mean absolute sensitivity normalized by sample size
  abs_sens_nn      <- lapply(sens_nn, abs)
  mean_abs_sens_nn <- as.data.frame(apply(simplify2array(abs_sens_nn), 1:2, mean) / numerosity[ii])
  
  colnames(mean_abs_sens_nn) <- c("Autosynth", "PCA", "AMPI", "Mean")
  
  df_long_sens <- mean_abs_sens_nn %>% 
    pivot_longer(
      cols      = everything(),
      names_to  = "Method",
      values_to = "Value"
    )
  
  df_long_sens$Method <- as.factor(df_long_sens$Method)
  
  # Individual plot for each sample size
  g_sens_list[[ii]] <- ggplot(df_long_sens, aes(x = Method, y = Value, fill = Method)) +
    geom_boxplot() +
    ylim(0, 1) +
    scale_fill_manual(values = c("#CC79A7", "#56B4E9", "#009E73", "#E69F00")) +
    labs(
      x     = "Method", 
      y     = expression(rho), 
      title = paste0("N = ", numerosity[[ii]])
    ) +
    theme_minimal() + 
    theme(
      axis.title.x = element_text(size = 9),
      axis.title.y = element_text(size = 9)
    )
}