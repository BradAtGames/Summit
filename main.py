'''
Driver code for TextRank application. Run with -h for usage information.
'''
import sys
import getopt
from summit.summarizer import summarize


def get_arguments():
    '''
    Read command line arguments
    '''
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "t:r:h", [
            "text=", "ratio=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(1)

    path = None
    ratio = 0.3
    words = None

    for opt, arg in opts:
        if opt in ("-t", "--text"):
            path = arg
        elif opt in ("-r", "--ratio"):
            ratio = float(arg)
        elif opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        else:
            assert False, str.format("unhandled option {0}", opt)

    return path, ratio


HELP_TEXT = """Usage: python main.py -t FILE [-r RATIO]  [-w COUNT] [-h]
\t-t FILE, --text=FILE:
\t\tPATH to text to summarize
\t-r RATIO, --ratio=RATIO:
\t\tFloat number (0,1] that defines the length of the summation as a proportion of original text length.
\t-w WORDS, --word-count=COUNT:
\t\tA number to limit the length of the summary. Ratio is ignored if word limit is set.abs
\t-h, --help:
\t\tPrint this help  message
"""


def usage():
    '''
    Print usage information
    '''
    print(HELP_TEXT)


def main():
    '''
    Main program entrypoint
    '''
    path, ratio = get_arguments()

    with open(path) as file:
        text = file.read()

    print("Original Text:")
    print(text)
    print()
    print("Summarized Text:")
    print(summarize(text, ratio))


if __name__ == "__main__":
    main()
