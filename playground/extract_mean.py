import rpy2.robjects as robjects
import argparse

def extract_from_logfile(logfile):
    robjects.r('library(irace)')
    get_mean = robjects.r('''function (x) {
        ireaceResults <- read_logfile(x);
        mean(ireaceResults$testing$experiment)
    }''')
    return float(get_mean(logfile)[0])

def main():
    parser = argparse.ArgumentParser(description='Extract mean from irace logfile.')
    parser.add_argument('logfile', type=str, help='irace logfile')
    args = parser.parse_args()
    print(extract_from_logfile(args.logfile))

if __name__ == '__main__':
    main()