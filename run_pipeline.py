"""
Master Pipeline Script
======================

This script runs the entire meta-analysis pipeline from data loading
to manuscript materials assembly.

Execution Order:
1. Data Loading (01_data_loading.py)
2. Data Cleaning (02_data_cleaning.py)
3. Descriptor Enrichment (03_descriptor_enrichment.py)
4. Exploratory Analysis (04_exploratory_analysis.py)
5. Model Fitting (05_model_fitting.py)
6. Model Diagnostics (06_model_diagnostics.py)
7. Sensitivity Analysis (07_sensitivity_analysis.py)
8. Environmental Module (08_environmental_module.py)
9. Evidence-Based Manuscript Assembly (09_manuscript_materials.py)

Usage:
    python run_pipeline.py [--start STEP] [--end STEP] [--skip STEPS]

Examples:
    python run_pipeline.py                    # Run all steps
    python run_pipeline.py --start 5          # Start from step 5
    python run_pipeline.py --end 4            # Run only steps 1-4
    python run_pipeline.py --skip 5,6         # Skip steps 5 and 6

Author: Manuscript authors
Date: March 2026
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))
from utils import ProjectConfig, setup_logging, print_section_header


# Define pipeline steps
PIPELINE_STEPS = [
    {
        'id': 1,
        'name': 'Data Loading',
        'script': 'scripts/01_data_loading.py',
        'required': True
    },
    {
        'id': 2,
        'name': 'Data Cleaning',
        'script': 'scripts/02_data_cleaning.py',
        'required': True
    },
    {
        'id': 3,
        'name': 'Descriptor Enrichment',
        'script': 'scripts/03_descriptor_enrichment.py',
        'required': True
    },
    {
        'id': 4,
        'name': 'Exploratory Analysis',
        'script': 'scripts/04_exploratory_analysis.py',
        'required': False
    },
    {
        'id': 5,
        'name': 'Model Fitting',
        'script': 'scripts/05_model_fitting.py',
        'required': True
    },
    {
        'id': 6,
        'name': 'Model Diagnostics',
        'script': 'scripts/06_model_diagnostics.py',
        'required': False
    },
    {
        'id': 7,
        'name': 'Sensitivity Analysis',
        'script': 'scripts/07_sensitivity_analysis.py',
        'required': False
    },
    {
        'id': 8,
        'name': 'Environmental Module',
        'script': 'scripts/08_environmental_module.py',
        'required': False
    },
    {
        'id': 9,
        'name': 'Manuscript Materials Assembly',
        'script': 'scripts/09_manuscript_materials.py',
        'required': False
    }
]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the complete Qm meta-analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Start from this step (default: 1)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=9,
        help='End at this step (default: 9)'
    )
    
    parser.add_argument(
        '--skip',
        type=str,
        default='',
        help='Comma-separated list of steps to skip (e.g., "6,7")'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )
    
    return parser.parse_args()


def run_step(step: dict, logger: logging.Logger, dry_run: bool = False) -> int:
    """
    Run a single pipeline step.
    
    Parameters
    ----------
    step : dict
        Step specification
    logger : logging.Logger
        Logger instance
    dry_run : bool
        If True, only show what would be executed
        
    Returns
    -------
    int
        Exit code (0 for success)
    """
    step_id = step['id']
    step_name = step['name']
    script_path = Path(step['script'])
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP {step_id}: {step_name.upper()}")
    logger.info(f"Script: {script_path}")
    logger.info(f"{'='*80}\n")
    
    if dry_run:
        logger.info("[DRY RUN] Would execute this step")
        return 0
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return 1
    
    # Execute script
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log output
        if result.stdout:
            logger.info(result.stdout)
        
        if result.returncode == 0:
            logger.info(f"\n[OK] Step {step_id} completed successfully in {duration:.1f}s")
            return 0
        else:
            logger.error(f"\n[FAIL] Step {step_id} failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")
            return result.returncode
            
    except subprocess.TimeoutExpired:
        logger.error(f"\n[FAIL] Step {step_id} timed out after 1 hour")
        return 1
        
    except Exception as e:
        logger.error(f"\n[FAIL] Step {step_id} failed with exception: {e}")
        return 1


def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Parse skip list
    skip_steps = set()
    if args.skip:
        try:
            skip_steps = {int(s.strip()) for s in args.skip.split(',')}
        except ValueError:
            print(f"ERROR: Invalid skip list: {args.skip}")
            return 1
    
    # Initialize logging
    try:
        config = ProjectConfig()
        logger = setup_logging(config, 'run_pipeline')
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        return 1
    
    # Print pipeline summary
    print_section_header("Qm META-ANALYSIS PIPELINE", logger=logger)
    
    logger.info(f"Configuration:")
    logger.info(f"  Start step: {args.start}")
    logger.info(f"  End step: {args.end}")
    logger.info(f"  Skip steps: {skip_steps if skip_steps else 'None'}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Random seed: {config.get('random_seed')}")
    logger.info(f"  Project version: {config.get('version')}")
    
    # Filter steps to execute
    steps_to_run = [
        step for step in PIPELINE_STEPS
        if args.start <= step['id'] <= args.end
        and step['id'] not in skip_steps
    ]
    
    logger.info(f"\nSteps to execute: {len(steps_to_run)}")
    for step in steps_to_run:
        required_str = " (REQUIRED)" if step['required'] else ""
        logger.info(f"  {step['id']}. {step['name']}{required_str}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN MODE] No scripts will be executed")
    
    # Execute pipeline
    start_time = datetime.now()
    failed_steps = []
    
    for step in steps_to_run:
        exit_code = run_step(step, logger, dry_run=args.dry_run)
        
        if exit_code != 0:
            failed_steps.append(step['name'])
            if step['required']:
                logger.error(f"\n[FAIL] PIPELINE FAILED: Required step '{step['name']}' failed")
                logger.error(f"Cannot continue with remaining steps")
                return 1
            else:
                logger.warning(f"\nStep '{step['name']}' failed but is not required. Continuing...")
    
    # Summary
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    logger.info(f"Steps executed: {len(steps_to_run)}")
    logger.info(f"Steps failed: {len(failed_steps)}")
    
    if failed_steps:
        logger.warning(f"\nFailed steps:")
        for step_name in failed_steps:
            logger.warning(f"  - {step_name}")
    
    if not failed_steps:
        logger.info(f"\n[OK] PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"\nResults available in:")
        logger.info(f"  Data: {config.get_path('processed_data')}")
        logger.info(f"  Models: {config.get_path('models')}")
        logger.info(f"  Results: {config.get_path('results')}")
        logger.info(f"  Figures: {config.get_path('figures')}")
        logger.info(f"  Logs: {config.get_path('logs')}")
        return 0
    else:
        logger.error(f"\n[WARN] PIPELINE COMPLETED WITH WARNINGS")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
