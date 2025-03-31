from ersilia import ErsiliaModel
import os
import pandas as pd

def featurize_data(model_id, datasets):
    model = ErsiliaModel(model=model_id)
    model.serve()

    for input_file, output_file in datasets.items():
        if os.path.exists(input_file):
            # Load original dataset to extract labels
            original_data = pd.read_csv(input_file)

            # Store the target column
            target = original_data["Y"]

            # Run featurization
            model.run(input=input_file, output=output_file)

            # Load featurized data
            featurized_data = pd.read_csv(output_file)

            # Add the target column back
            featurized_data["Y"] = target

            # Save the final dataset with labels
            featurized_data.to_csv(output_file, index=False)

        else:
            raise FileNotFoundError(f"Input file '{input_file}' not found!")

if __name__ == "__main__":
    model_id = "eos8a4x"

    datasets = {
        "data/bbbp_train.csv": "data/bbbp_train_featurised.csv",
        "data/bbbp_test.csv": "data/bbbp_test_featurised.csv",
        "data/bbbp_valid.csv": "data/bbbp_valid_featurised.csv",
    }

    featurize_data(model_id, datasets)
