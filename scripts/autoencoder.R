################################################################################
# FUNCTION: AUTOENCODER BUILDER & TRAINER
# 
# Description:
#   Constructs and trains a deep autoencoder using the Keras API.
#
# Arguments:
#   input_dim  : (Integer) Number of features in the input data.
#   latent_dim : (Integer) Dimension of the bottleneck (compressed) layer.
#   dt         : (Matrix/Array) The input dataset for training.
#   iter       : (Integer) Number of training epochs.
#   layers     : (Integer) Number of neurons in the intermediate dense layers.
#   batch      : (Integer) Size of the training batches.
#   validation : (Float) Fraction of the data to be used for validation (0 to 1).
#   activation : (String) Activation function to use (e.g., 'relu', 'tanh', 'sigmoid').
#
# Returns:
#   A list containing:
#     - encoder     : The trained encoder model (Input -> Latent).
#     - autoencoder : The full trained model (Input -> Output).
#     - history     : The training history object (logs, loss, etc.).
################################################################################

autoencoder <- function(input_dim, 
                        latent_dim, 
                        dt, 
                        iter, 
                        layers, 
                        batch, 
                        validation, 
                        activation) {
  
  # ---- 1. PREPROCESSING ----
  
  # Ensure the input data (dt) is in the correct format (matrix or array)
  dt <- as.matrix(dt)
  
  # ---- 2. ARCHITECTURE DEFINITION ----
  
  # Define the input layer with the specified dimensions
  input_layer <- layer_input(shape = c(input_dim))
  
  # ENCODER SECTION
  # Compresses the input into a lower-dimensional latent space
  encoded <- input_layer %>%
    layer_dense(units = layers, activation = activation) %>%
    layer_dense(units = latent_dim, activation = activation)
  
  # DECODER SECTION
  # Reconstructs the original input from the latent representation
  decoded <- encoded %>%
    layer_dense(units = layers, activation = activation) %>%
    layer_dense(units = input_dim, activation = activation)
  
  # FULL AUTOENCODER MODEL
  # Combines encoder and decoder into a single end-to-step model
  autoencoder_model <- keras_model(inputs = input_layer, outputs = decoded)
  
  # ---- 3. COMPILATION ----
  
  # Configure the model for training
  # Optimizer: Adam | Loss: Mean Squared Error (standard for reconstruction tasks)
  autoencoder_model %>% compile(
    optimizer = optimizer_adam(),
    loss      = loss_mean_squared_error()
  )
  
  # ---- 4. TRAINING ----
  
  # Train the model to reconstruct its own input (X = dt, Y = dt)
  history <- autoencoder_model %>% fit(
    x                = dt,  
    y                = dt,  
    epochs           = iter,
    batch_size       = batch,
    validation_split = validation
  )
  
  # ---- 5. OUTPUT EXTRACTION ----
  
  # Define a separate model for the encoder to extract latent features later
  encoder_model <- keras_model(inputs = input_layer, outputs = encoded)
  
  # Return a list containing the specialized models and the training logs
  return(list(
    encoder     = encoder_model, 
    autoencoder = autoencoder_model, 
    history     = history
  ))
}

# ---- EXAMPLE USAGE (Commented) ----

# # 1. Prepare dummy data (1000 samples, 20 features)
# my_data <- matrix(runif(1000 * 20), ncol = 20)
#
# # 2. Run the autoencoder function
# model_results <- autoencoder(
#   input_dim  = 20,
#   latent_dim = 5,
#   dt         = my_data,
#   iter       = 50,
#   layers     = 12,
#   batch      = 32,
#   validation = 0.2,
#   activation = "relu"
# )
#
# # 3. Access the results
# # summary(model_results$autoencoder)        # View model summary
# # plot(model_results$history)               # Plot training loss
# # latent_features <- predict(model_results$encoder, my_data) # Get compressed data