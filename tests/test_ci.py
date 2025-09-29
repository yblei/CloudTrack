import unittest
import sys
import subprocess


class TestCLI(unittest.TestCase):
    """Test cases for the CloudTrack CLI functionality."""

    def test_cli_help_output(self):
        """Test that 'python -m cloud_track --help' returns valid output."""
        try:
            # Run the CLI help command
            result = subprocess.run(
                [sys.executable, "-m", "cloud_track", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check that the command executed successfully
            self.assertEqual(result.returncode, 0, 
                           f"CLI command failed with stderr: {result.stderr}")
            
            # Check that we got some output
            self.assertGreater(len(result.stdout), 0, 
                             "CLI help command produced no output")
            
            # Check for basic help content
            self.assertIn("Usage:", result.stdout, 
                        "Help output should contain 'Usage:'")
            
        except subprocess.TimeoutExpired:
            self.fail("CLI help command timed out after 30 seconds")
        except Exception as e:
            self.fail(f"Failed to run CLI help command: {e}")

    def test_cli_module_import(self):
        """Test that the CLI module can be imported successfully."""
        try:
            from cloud_track import cli
            self.assertIsNotNone(cli)
        except ImportError as e:
            self.fail(f"Could not import cloud_track.cli: {e}")

    def test_cli_main_import(self):
        """Test that the CLI main module can be imported successfully."""
        try:
            import cloud_track.__main__
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"Could not import cloud_track.__main__: {e}")


if __name__ == "__main__":
    unittest.main()
