from src.evaluate import evaluate_models


def main():
    print("=" * 60)
    print(" AI-Powered Fake News Detection Using Text Classification ")
    print("=" * 60)

    print("\nTraining and evaluating models...\n")

    results = evaluate_models()

    print("\nTraining completed successfully.\n")
    print(results)

    print("\nProject outputs generated successfully:")
    print("• Trained models")
    print("• Model comparison CSV")
    print("• Classification report")
    print("• Best model information")
    print("• Predictions CSV")
    print("• Confusion matrices")
    print("• Performance graphs")

    print("\nRun the Flask application using:")
    print("python app.py")


if __name__ == "__main__":
    main()