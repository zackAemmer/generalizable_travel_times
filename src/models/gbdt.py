
# # Resample GPS points to fixed number
# kcm_train_data_resample = data_utils.resample_deeptte_gps(kcm_train_data, 128)
# kcm_test_data_resample = data_utils.resample_deeptte_gps(kcm_test_data, 128)

# atb_train_data_resample = data_utils.resample_deeptte_gps(atb_train_data, 128)
# atb_test_data_resample = data_utils.resample_deeptte_gps(atb_test_data, 128)

# # Reshape the resampled GPS data to a 2d np array for train/testing additional models
# X_train_kcm, y_train_kcm = data_utils.format_deeptte_to_features(kcm_train_data, kcm_train_data_resample)
# X_test_kcm, y_test_kcm = data_utils.format_deeptte_to_features(kcm_test_data, kcm_test_data_resample)

# X_train_atb, y_train_atb = data_utils.format_deeptte_to_features(atb_train_data, atb_train_data_resample)
# X_test_atb, y_test_atb = data_utils.format_deeptte_to_features(atb_test_data, atb_test_data_resample)

# # Train GBDT on training data, make preds on test data
# kcm_reg = GradientBoostingRegressor(random_state=0)
# kcm_reg.fit(X_train_kcm, y_train_kcm)
# GradientBoostingRegressor(random_state=0)
# kcm_gbdt_preds = kcm_reg.predict(X_test_kcm)

# atb_reg = GradientBoostingRegressor(random_state=0)
# atb_reg.fit(X_train_atb, y_train_atb)
# GradientBoostingRegressor(random_state=0)
# atb_gbdt_preds = atb_reg.predict(X_test_atb)