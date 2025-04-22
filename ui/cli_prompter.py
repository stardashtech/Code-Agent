import logging

logger = logging.getLogger(__name__)

def ui_confirm(prompt: str) -> bool:
    """ 
    Prompts the user for a yes/no confirmation in the CLI.

    Args:
        prompt: The question to ask the user.

    Returns:
        True if the user confirms (e.g., enters 'yes', 'y'), False otherwise.
    """
    while True:
        try:
            # Print prompt to stderr to avoid interfering with potential stdout parsing
            # of the main application.
            print(f"\nCONFIRMATION REQUIRED:\n{prompt}", file=sys.stderr) 
            response = input("(yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                logger.info("User confirmed via CLI.")
                return True
            elif response in ['no', 'n']:
                logger.warning("User denied via CLI.")
                return False
            else:
                print("Invalid input. Please enter 'yes' or 'no'.", file=sys.stderr)
        except EOFError:
             # Handle cases where input stream is closed (e.g., non-interactive environment)
             logger.warning("EOFError encountered during UI confirmation. Defaulting to NO.")
             return False
        except Exception as e:
            logger.error(f"Error during UI confirmation: {e}. Defaulting to NO.", exc_info=True)
            return False

# Add sys import needed for stderr
import sys 