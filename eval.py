import argparse
from evaluator import Evaluator, MutaExplainEvaluator, MutaEngineerEvaluator, FitnessOptimizeEvaluator

def add_arguments(parser):
    parser.add_argument("--muta_explain", action="store_true")
    parser.add_argument("--muta_engineer", action="store_true")
    parser.add_argument("--fitness_optimize", action="store_true")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_known_args()[0]

    if args.muta_explain:
        cls = MutaExplainEvaluator
    elif args.muta_engineer:
        cls = MutaEngineerEvaluator
    if args.fitness_optimize:
        cls = FitnessOptimizeEvaluator

    parser = cls.add_arguments(parser)
    args = parser.parse_args()
    evaluator = cls(args)
    evaluator.evaluate()