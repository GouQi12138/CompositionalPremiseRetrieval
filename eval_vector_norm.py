import argparse
import torch
import sys

from sentence_transformers import SentenceTransformer

from train_composition import gen_projection_model
from evaluation.eval_baseline_model import genL2report

def evaluateCompositional(model, query_model):
    # Load data
    #test_dataset = TripletEvalDataset(os.path.join(os.getcwd(), args.test_data))
    test_dataset = TripletEvalDataset(os.path.join(os.getcwd(), 'data/test_data.txt'))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate
    evaluator = RerankingEvaluator(test_dataset)
    query_model.evaluate(evaluator, output_path=args.save_dir)
    #query_model.evaluate(evaluator, output_path='output/')

    # Test
    query_model = SentenceTransformer(args.save_dir).to(device)
    evaluateCompositional(query_model, query_model)



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #./checkpoints/triplet_adjacent/
    model = SentenceTransformer(args.model).to(device)
    if args.model_path:
        print("Loading query model from checkpoint...")
        model.load_state_dict(torch.load(args.model_path))

    if args.debug:
        print(model)
    
    # evaluate(query_model, query_model, split=args.split, debug=args.debug)
    # retrieve(query_model)

    # remove the last Normalization layer
    #query_model = SentenceTransformer(modules=[query_model[0], query_model[1]]).to(device)

    # Projection model
    #model = gen_projection_model(model, device)

    # L2 norm of premise-pool, test data from 3 baseline models
    genL2report(model, args.fig_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="pre-trained model \
                            (best general purpose model: all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2); \
                            best semantic search model: multi-qa-mpnet-base-dot-v1 (https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1))")
    parser.add_argument("--model-path", type=str, help="specify to evaluate a specific query model")
    parser.add_argument("--split", type=str, default="test", help="data split to evaluate on (train, dev, test)")
    parser.add_argument("--fig-filename", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
