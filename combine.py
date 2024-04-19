import os

def combine_reports(reports_dir):
    """
    Combines detailed analysis reports from subdirectories within a reports directory
    into a single file in the parent directory.

    Args:
        reports_dir (str): The path to the directory containing subdirectories with reports.
    """

    detailed_report_path = os.path.join(reports_dir, "detailed_report.txt")

    with open(detailed_report_path, 'w') as outfile:
        outfile.write("Combined Detailed Report\n")  # Add the first line

        for dirpath, dirnames, filenames in os.walk(reports_dir):
            for filename in filenames:
                if filename == "detailed_analysis_report.txt":
                    # Get the directory name for titling
                    title = os.path.basename(dirpath)  

                    # Write title and separator
                    outfile.write(f"\n{title}\n-----------------\n")

                    # Read and write report contents
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, 'r') as infile:
                        outfile.write(infile.read())

                    outfile.write("\n\n")  # Add spacing between reports

    print("Reports combined into detailed_report.txt")


# reports_directory = "results\20240418-205459"
# combine_reports(reports_directory)
