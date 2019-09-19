def missing_values(X, X_test_full):

    print('Missing train data? ', X.isnull().any().any())
    print('Missing test data? ', X_test_full.isnull().any().any())
    print("Shape train: ", X.shape)
    print("Shape test: ", X_test_full.shape)

    return 0
