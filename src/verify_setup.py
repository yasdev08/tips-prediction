"""
Tips Prediction Project - Setup Verification Script
Run this script to verify all dependencies and files are properly set up
"""

import sys
import os


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(status, message):
    """Print status message"""
    icon = "✓" if status else "✗"
    color_start = "\033[92m" if status else "\033[91m"  # Green or Red
    color_end = "\033[0m"
    print(f"{color_start}{icon}{color_end} {message}")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("CHECKING PYTHON VERSION")
    version = sys.version_info
    required_version = (3, 7)

    current = f"{version.major}.{version.minor}.{version.micro}"
    is_compatible = (version.major, version.minor) >= required_version

    print(f"Current Python version: {current}")
    print(f"Required: Python {required_version[0]}.{required_version[1]}+")
    print_status(is_compatible, "Python version is compatible" if is_compatible
    else f"Python version too old! Please upgrade to {required_version[0]}.{required_version[1]}+")

    return is_compatible


def check_required_packages():
    """Check if all required packages are installed"""
    print_header("CHECKING REQUIRED PACKAGES")

    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib.pyplot',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'gradio': 'gradio',
    }

    all_installed = True
    installed_versions = {}

    for package, install_name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')

            print_status(True, f"{install_name:20s} version {version}")
            installed_versions[package] = version
        except ImportError:
            print_status(False, f"{install_name:20s} NOT INSTALLED")
            all_installed = False

    if not all_installed:
        print("\n⚠️  Some packages are missing!")
        print("Install missing packages with:")
        print("pip install pandas numpy matplotlib seaborn scikit-learn gradio")
    else:
        print("\n✓ All required packages are installed!")

    return all_installed, installed_versions


def check_dataset():
    """Check if dataset file exists and is valid"""
    print_header("CHECKING DATASET")

    dataset_path = 'tips.csv'

    # Check file exists
    if not os.path.exists(dataset_path):
        print_status(False, f"Dataset file '{dataset_path}' not found!")
        print("\n⚠️  Please ensure tips.csv is in the same directory as this script.")
        return False

    print_status(True, f"Dataset file '{dataset_path}' found")

    # Try to load and validate dataset
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)

        # Check shape
        expected_rows = 244
        expected_cols = 7

        print(f"  • Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print_status(df.shape[0] == expected_rows and df.shape[1] == expected_cols,
                     f"Expected: {expected_rows} rows × {expected_cols} columns")

        # Check columns
        expected_columns = ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
        missing_columns = set(expected_columns) - set(df.columns)

        if missing_columns:
            print_status(False, f"Missing columns: {missing_columns}")
            return False

        print_status(True, "All expected columns present")

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        print_status(missing_count == 0,
                     f"Missing values: {missing_count} (should be 0)")

        # Basic statistics
        print(f"\n  Dataset Statistics:")
        print(f"    • Total Bill Range: ${df['total_bill'].min():.2f} - ${df['total_bill'].max():.2f}")
        print(f"    • Tip Range: ${df['tip'].min():.2f} - ${df['tip'].max():.2f}")
        print(f"    • Average Tip: ${df['tip'].mean():.2f}")

        return True

    except Exception as e:
        print_status(False, f"Error loading dataset: {str(e)}")
        return False


def check_model_files():
    """Check if model files exist"""
    print_header("CHECKING MODEL FILES")

    model_files = [
        'linear_regression_model.pkl',
        'decision_tree_model.pkl',
        'random_forest_model.pkl',
        'gradient_boosting_model.pkl',
        'scaler.pkl',
        'label_encoders.pkl'
    ]

    found_models = []
    missing_models = []

    for model_file in model_files:
        exists = os.path.exists(model_file)
        if exists:
            found_models.append(model_file)
            print_status(True, f"{model_file}")
        else:
            missing_models.append(model_file)
            print_status(False, f"{model_file} (not found)")

    if missing_models:
        print(f"\n⚠️  {len(missing_models)} model file(s) missing")
        print("Models will be created when you run the Jupyter notebook.")
        print("This is normal if you haven't run the notebook yet.")
    else:
        print(f"\n✓ All {len(found_models)} model files found!")
        print("You can now run the Gradio interface.")

    return len(found_models), len(missing_models)


def check_project_files():
    """Check if key project files exist"""
    print_header("CHECKING PROJECT FILES")

    project_files = {
        'tips_prediction.py': 'Jupyter notebook (analysis)',
        'tips_gradio_app.py': 'Gradio web interface',
        'tips.csv': 'Dataset',
    }

    all_present = True

    for filename, description in project_files.items():
        exists = os.path.exists(filename)
        print_status(exists, f"{filename:30s} - {description}")
        if not exists:
            all_present = False

    if not all_present:
        print("\n⚠️  Some project files are missing!")
        print("Please ensure all files are in the same directory.")
    else:
        print("\n✓ All project files present!")

    return all_present


def test_model_prediction():
    """Test if models can make predictions"""
    print_header("TESTING MODEL PREDICTION")

    # Check if Random Forest model exists
    model_path = 'linear_regression_model.pkl'

    if not os.path.exists(model_path):
        print_status(False, "Random Forest model not found - skipping test")
        print("Run the Jupyter notebook first to train models.")
        return False

    try:
        import pickle
        import numpy as np

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print_status(True, "Model loaded successfully")

        # Test prediction
        test_features = np.array([[25.0, 2, 1, 0, 1, 0]])  # Example input
        prediction = model.predict(test_features)[0]

        print_status(True, f"Test prediction: ${prediction:.2f}")
        print(f"  • Test input: Bill=$25, Size=2, Male, Non-smoker, Saturday, Dinner")

        # Validate prediction
        if 0 < prediction < 20:  # Reasonable tip range
            print_status(True, "Prediction is in reasonable range")
            return True
        else:
            print_status(False, "Prediction seems unusual")
            return False

    except Exception as e:
        print_status(False, f"Error testing model: {str(e)}")
        return False


def print_next_steps(all_checks_passed):
    """Print next steps based on verification results"""
    print_header("NEXT STEPS")

    if all_checks_passed:
        print("✅ All verifications passed! Your setup is complete.")
        print("\n📝 Recommended workflow:")
        print("  1. Open and run: tips_prediction.py")
        print("     → This will train models and generate visualizations")
        print("  2. Run: python tips_gradio_app.py")
        print("     → This will launch the web interface")


        print("\n🚀 Quick start commands:")
        print("  jupyter notebook tips_prediction.py")
        print("  python tips_gradio_app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("  • Install missing packages:")
        print("    pip install pandas numpy matplotlib seaborn scikit-learn gradio")
        print("  • Ensure tips.csv is in the current directory")
        print("  • Make sure you're using Python 3.7+")


def main():
    """Main verification function"""
    print("\n" + "=" * 70)
    print(" " * 15 + "TIPS PREDICTION PROJECT")
    print(" " * 15 + "Setup Verification Script")
    print("=" * 70)

    # Run all checks
    checks = {}

    checks['python'] = check_python_version()
    checks['packages'], installed = check_required_packages()
    checks['dataset'] = check_dataset()
    checks['project_files'] = check_project_files()

    # Model files check (not required for initial setup)
    found, missing = check_model_files()
    checks['models_ready'] = (found > 0)

    # Optional: Test model if available
    if checks['models_ready']:
        checks['prediction'] = test_model_prediction()

    # Summary
    print_header("VERIFICATION SUMMARY")

    critical_checks = ['python', 'packages', 'dataset', 'project_files']
    critical_passed = all(checks.get(check, False) for check in critical_checks)

    print(f"\nCritical Requirements:")
    for check in critical_checks:
        status = "✓ PASS" if checks.get(check, False) else "✗ FAIL"
        print(f"  {status} - {check}")

    print(f"\nOptional Components:")
    print(f"  {'✓ READY' if checks.get('models_ready', False) else '⚠ PENDING'} - Trained models")
    if checks.get('prediction'):
        print(f"  ✓ WORKING - Model predictions")

    # Print next steps
    print_next_steps(critical_passed)

    # Return status
    if critical_passed:
        print("\n" + "=" * 70)
        print("  🎉 SETUP VERIFIED - You're ready to start!")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n" + "=" * 70)
        print("  ⚠️  SETUP INCOMPLETE - Please fix issues above")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)