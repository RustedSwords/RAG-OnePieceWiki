from utils import client, generate_embeddings, generate_response, COLLECTION_NAME

def main():
    prompt = input("Enter a prompt: ")

    query_prompt = f"Represent this sentence for searching relevant passages: {prompt}"
    embedding = generate_embeddings(query_prompt)

    if embedding is None:
        print("Failed to generate query embedding")
        return

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        with_payload=True,
        limit=5,
    )

    passages = "\n".join(
        f"- {point.payload['text']}"
        for point in results.points
    )

    augmented_prompt = f"""
The following are the relevant passages:
<retrieved-data>
{passages}
</retrieved-data>

Answer the following question using the retrieved data:
<user-prompt>
{prompt}
</user-prompt>
"""

    response = generate_response(augmented_prompt)
    print(response)


if __name__ == "__main__":
    main()