import argparse
import logging
from pathlib import Path

from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from epbench.src.evaluation.evaluation_wrapper import EvaluationWrapper
from epbench.src.evaluation.precomputed_results import get_precomputed_results
from epbench.src.results.average_groups import extract_groups
import asyncio

async def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting quickstart.py script")
    logger.info("=" * 80)

    git_repo_filepath = '/Users/jinggong/memmachine/episodic-memory-benchmark'
    data_folder = Path(git_repo_filepath) / 'epbench' / 'data'
    env_file = Path(git_repo_filepath) / '.env'
    logger.info("Default paths initialized:")
    logger.info(f"  - Git repo path: {git_repo_filepath}")
    logger.info(f"  - Data folder: {data_folder}")
    logger.info(f"  - Env file: {env_file}")

    # Parsing the arguments
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default=str(data_folder),
                        help='Path to the data folder')
    parser.add_argument('--env_file', type=str, default=str(env_file),
                        help='Path to the .env file')
    parser.add_argument('--book_nb_events', type=int, default=20,
                        help='Number of events in the book (20 for short and 200 for long, for the default experiment)')
    parser.add_argument('--answering_kind', type=str, default='prompting',
                        help='Answering kind')
    parser.add_argument('--answering_model_name', type=str, default='gpt-4o-mini-2024-07-18',
                        help='Answering model name')
    parser.add_argument('--memmachine_ingest', action='store_true',
                        help='Flag to indicate whether to ingest into MemMachine')

    # Overrid the file paths
    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    env_file = Path(args.env_file)
    logger.info("Arguments parsed successfully:")
    logger.info(f"  - data_folder: {data_folder}")
    logger.info(f"  - env_file: {env_file}")
    logger.info(f"  - book_nb_events: {args.book_nb_events}")
    logger.info(f"  - answering_kind: {args.answering_kind}")
    logger.info(f"  - answering_model_name: {args.answering_model_name}")
    logger.info(f"  - memmachine_ingest: {args.memmachine_ingest}")

    # Step 1: generating the synthetic episodic memory dataset
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 1: Generating the synthetic episodic memory dataset")
    logger.info("=" * 80)

    # Configuration (here, default short book with 20 events)
    logger.info("Configuring parameters for benchmark generation...")
    book_parameters = {
        'indexing': 'default',
        'nb_summaries': 0
    }
    prompt_parameters = {
        'nb_events': args.book_nb_events,
        'name_universe': 'default',
        'name_styles': 'default',
        'seed': 0,
        'distribution_events': {
            'name': 'geometric',
            'param': 0.1
        }
    }
    model_parameters = {
        'model_name': 'gpt-4o-mini-2024-07-18',
        'max_new_tokens': 4096,
        'itermax': 10
    }
    logger.info("Configuration parameters:")
    logger.info(f"  - book_parameters: {book_parameters}")
    logger.info(f"  - prompt_parameters: {prompt_parameters}")
    logger.info(f"  - model_parameters: {model_parameters}")

    # Generation (generate the book, then compute the ground truth QAs)
    logger.info("")
    logger.info("Creating BenchmarkGenerationWrapper and generating benchmark...")
    my_benchmark = BenchmarkGenerationWrapper()
    await my_benchmark.init(
        prompt_parameters, model_parameters, book_parameters, data_folder, env_file, memmachine_ingest=args.memmachine_ingest
    )
    logger.info("✓ Benchmark generation completed")
    logger.info(f"  - Benchmark object created: {type(my_benchmark).__name__}")

    # Step 2: predicting the answers given the document and the questions
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: Predicting answers given the document and questions")
    logger.info("=" * 80)

    # Configuration
    logger.info("Configuring answering parameters...")
    answering_parameters = {
        'kind': args.answering_kind,
        'model_name': args.answering_model_name,
        'max_new_tokens': 4096,
        'sleeping_time': 0,
        'policy': 'remove_duplicates'
    }
    logger.info(f"Answering parameters: {answering_parameters}")

    # Prediction (generate answers, then evaluate them)
    logger.info("")
    logger.info("Creating EvaluationWrapper and generating predictions...")
    my_evaluation = EvaluationWrapper()
    await my_evaluation.init(my_benchmark, answering_parameters, data_folder, env_file)
    logger.info("✓ Answer prediction and evaluation completed")
    logger.info(f"  - Evaluation object created: {type(my_evaluation).__name__}")

    # Step 3: extract the performance results
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 3: Extracting performance results")
    logger.info("=" * 80)

    # Configuration
    logger.info("Configuring experiments and benchmarks for results extraction...")
    # Use the model name from model_parameters for book_model_name
    book_model_name = model_parameters['model_name']
    experiments = [{
        'book_nb_events': args.book_nb_events,
        'book_model_name': book_model_name,
        'answering_kind': args.answering_kind,
        'answering_model_name': args.answering_model_name,
        'answering_embedding_chunk': 'n/a'
    }]
    # Create benchmark key based on model name and nb_events
    # NOTE: original (jgong)
    # all_benchmarks = {f'benchmark_claude_default_{args.book_nb_events}': my_benchmark}
    # NOTE: change to use the model name from model_parameters for book_model_name (jgong)
    # For gpt-4o-mini-2024-07-18, use a generic key that can be looked up
    benchmark_key = f'benchmark_{book_model_name}_default_{args.book_nb_events}'
    all_benchmarks = {benchmark_key: my_benchmark}
    logger.info(f"Experiments configuration: {experiments}")
    logger.info(f"Benchmarks dictionary keys: {list(all_benchmarks.keys())}")

    # Results
    logger.info("")
    logger.info("Loading precomputed results...")
    df = await get_precomputed_results(experiments, env_file, data_folder, all_benchmarks)
    logger.info(f"✓ Precomputed results loaded")
    logger.info(f"  - DataFrame shape: {df.shape}")
    logger.info(f"  - DataFrame columns: {list(df.columns)}")

    logger.info("")
    logger.info("Extracting and grouping results...")
    # select the book of interest (either 20 or 200)
    nb_events = args.book_nb_events
    # select the elements to group
    relative_to = ['get', 'bins_items_correct_answer']
    logger.info(f"  - Number of events: {nb_events}")
    logger.info(f"  - Grouping relative to: {relative_to}")
    # group the results according to `relative_to`
    df_results = extract_groups(df, nb_events, relative_to)
    logger.info(f"  - Grouped results shape: {df_results.shape}")
    # further filtering by selecting only the simple recall questions
    logger.info("Filtering results to select only simple recall questions (get == 'all')...")
    df_results = df_results[df_results['get'] == 'all'].drop('get', axis=1)
    logger.info(f"  - Filtered results shape: {df_results.shape}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"{df_results}")
    print(df_results)
    logger.info("")
    logger.info("=" * 80)
    logger.info("Script completed successfully!")
    logger.info("=" * 80)
    print('Ended successfully')

if __name__ == "__main__":
    asyncio.run(main())
