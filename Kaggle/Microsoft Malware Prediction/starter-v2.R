library(lightgbm)
library(data.table)

set.seed(3)

#---------------------------
cat("Loading data...\n")

dt <- fread("../input/train.csv", drop = "MachineIdentifier")
# dt <- dt[sample(.N, .N %/% 2), ]

y <- dt[, HasDetections]
dt[, HasDetections := NULL]
N <- dt[, .N]

dt <- rbind(dt, fread("../input/test.csv", drop = "MachineIdentifier"))

#---------------------------
cat("Adding features...\n")

dt[,'fe_non_prim_drive'] <- dt[,'Census_PrimaryDiskTotalCapacity'] - dt[,'Census_SystemVolumeTotalCapacity']
dt[,'fe_aspect_ratio'] = dt[,'Census_InternalPrimaryDisplayResolutionHorizontal'] / dt[,'Census_InternalPrimaryDisplayResolutionVertical']
dt[,'fe_screen_area'] = (dt[, 'fe_aspect_ratio'] * (dt[, 'Census_InternalPrimaryDiagonalDisplaySizeInInches']**2)) / (dt[, 'fe_aspect_ratio']**2 + 1)
dt[,'fe_monitor_dims'] = dt[,'Census_InternalPrimaryDisplayResolutionHorizontal'] * dt[,'Census_InternalPrimaryDisplayResolutionVertical']


dt[, f1 := .N / nrow(dt), by = "AvSigVersion,Wdft_IsGamer"
  ][, f2 := .N / nrow(dt), by = "Census_ProcessorCoreCount,Wdft_RegionIdentifier"
  ][, f3 := .N / nrow(dt), by = "Census_ProcessorCoreCount,Census_OEMNameIdentifier"
  ][, f4 := .N / nrow(dt), by = "GeoNameIdentifier,Census_OEMNameIdentifier"]

#---------------------------
cat("Converting character columns...\n")

cats <- names(which(sapply(dt, is.character)))
dt[, (cats) := lapply(.SD, function(x) as.integer(as.factor(x))), .SDcols = cats]  

dt <- data.matrix(dt)

rm(cats); invisible(gc())

#---------------------------
cat("Preparing data for boosting...\n")

tr <- lgb.Dataset(data = dt[1:N, ], label = y)
te <- dt[-(1:N), ]

rm(dt, y, N); invisible(gc())

#---------------------------
cat("Training model and predicting...\n")

subm <- fread("../input/sample_submission.csv")
subm[, HasDetections := 0]

n_bags <- 5
for (i in seq(0.7, 0.9, length.out = n_bags)) {
  cat("bag ", i, "\n")
  p <- list(boosting_type = "gbdt", 
            objective = "binary",
            metric = "auc", 
            learning_rate = 0.05, 
            max_depth = 5,
            num_leaves = 40,
            sub_feature = i, 
            sub_row = i, 
            bagging_freq = 1,
            lambda_l1 = 0.1, 
            lambda_l2 = 0.1)
  
  m_gbm <- lgb.train(p, tr, 5200, verbose = -1)
  subm[, HasDetections := HasDetections + predict(m_gbm, te) / n_bags]
  
  rm(m_gbm); invisible(gc())
}

#---------------------------
cat("Making submission file...\n")

fwrite(subm, "../output/rstarter-v2-fulldata.csv")