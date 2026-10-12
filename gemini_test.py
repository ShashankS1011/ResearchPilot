import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


def main():
    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in .env"
        )

    client = genai.Client(
        api_key=api_key
    )

    print(
        "Testing Gemini Interactions API..."
    )

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=(
            "Explain the photovoltaic effect "
            "in two sentences."
        ),
    )

    print(
        "\nGemini response:\n"
    )

    print(
        interaction.output_text
    )


if __name__ == "__main__":
    main()