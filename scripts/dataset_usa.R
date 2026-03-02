################################################################################
# DATASET PREPARATION: USA COUNTY DATA
# Purpose: Clean and normalize USA county data for AMPI analysis
################################################################################

# ---- 1. SETUP ----

# Install the data package if not already present
# devtools::install_github("deleetdk/USA.county.data")

library(USA.county.data)

# Load the raw dataset
us <- USA_county_data

# ---- 2. DATA SELECTION & PREPARATION ----

# Define the initial set of variables to keep
keep_initial <- c(
  "fips", "County", "Less.Than.High.School", "Graduate.Degree", "School.Enrollment",
  "Median.Earnings.2010.dollars", "White.not.Latino.Population",
  "African.American.Population", "Adults.65.and.Older.Living.in.Poverty",
  "Children.Under.6.Living.in.Poverty", "Poverty.Rate.below.federal.poverty.threshold",
  "Gini.Coefficient", "Poor.physical.health.days", "Poor.mental.health.days", 
  "Children.in.single.parent.households", "Adult.smoking", "Adult.obesity", 
  "Uninsured", "Unemployment", "Violent.crime", "Injury.deaths", "median_age", 
  "Low.birthweight", "Teen.births", "Diabetes", "Sexually.transmitted.infections", 
  "Homicide.rate", "Infant.mortality"
)

# Preliminary subset
yy <- us[, keep_initial]

# Refine selection for final analysis
keep_final <- c(
  "fips", "County", "Less.Than.High.School", "Graduate.Degree", "School.Enrollment",
  "Median.Earnings.2010.dollars", "White.not.Latino.Population",
  "African.American.Population", "Adults.65.and.Older.Living.in.Poverty",
  "Children.Under.6.Living.in.Poverty", "Poverty.Rate.below.federal.poverty.threshold",
  "Gini.Coefficient", "Children.in.single.parent.households",
  "Adult.obesity", "Uninsured", "Unemployment"
)

# Apply final selection and remove rows with missing values (NA)
yy <- yy[, keep_final]
yy <- na.omit(yy)

# Export processed raw data
raw_data <- yy
save(raw_data, file = "datasets/raw_data_usa.RData")

# ---- 3. AMPI NORMALIZATION & TRANSFORMATION ----

# Normalize data using the norm_ampi function (excluding non-numeric ID columns)
kk <- norm_ampi(as.matrix(yy[, -c(1:2)]))

# Invert specific indicators (Polarity adjustment)
# Note: Indicators are transformed using the formula: 200 - value
kk[, "Graduate.Degree"]                 <- 200 - kk[, "Graduate.Degree"]
kk[, "School.Enrollment"]               <- 200 - kk[, "School.Enrollment"]
kk[, "Median.Earnings.2010.dollars"]    <- 200 - kk[, "Median.Earnings.2010.dollars"]

# Polarity adjustment for White population 
# (Two steps maintained as per original script logic)
kk[, "White.not.Latino.Population"]     <- 200 - kk[, "White.not.Latino.Population"]

# Re-combine identifiers (fips, County) with normalized data
kk <- cbind(yy[, c(1:2)], as.matrix(kk))

# Export final normalized data
ampi_data <- kk
save(ampi_data, file = "datasets/normed_data_usa.RData")