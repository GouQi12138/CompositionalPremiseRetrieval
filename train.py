import argparse

from sentence_transformers import SentenceTransformer, util
import torch


def example_inference(model):
    # Follows https://www.sbert.net/docs/pretrained_models.html
    query_embedding = model.encode("How big is London")
    passage_embedding = model.encode(["London has 9,787,426 inhabitants at the 2011 census",
                                      "London is known for its financial district"])
    print("Similarity:", util.dot_score(query_embedding, passage_embedding))


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained SBERT
    model = SentenceTransformer(args.model).to(device)

    example_inference(model)
    
    # Load dataloader
    # Use SCL as loss function
    # Train SBERT as a single encoder (only query encoder is updated during fine-tuning)
    # Evaluate validation
    # Evaluate metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="pre-trained model \
                            (best general purpose model: all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2); \
                            best semantic search model: multi-qa-mpnet-base-dot-v1 (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1))")
    args = parser.parse_args()

    main(args)