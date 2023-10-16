import argparse
import json
import epm.webserver.flask_worker
import subprocess


def correct_switch(s: str):
    if s.startswith('-') and not s.startswith('--'):
        return '--' + s[1:]
    else:
        return s

def main(original_file, output_file, target_runner, do_correct_switch=True):
    with open(original_file, encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8') as wf:
            for line in f:
                o = json.loads(line)
                correct_time = o['time']
                seed = o['seed']
                instance = o['instance']
                config = [(k if not do_correct_switch else correct_switch(k), v) for k, v in o['config'].items()]
                new_args = [
                    '',
                    '',
                    str(seed),
                    '//' + instance, # We ignore add two slashes as a workaround.
                    '',
                    *[item for sublist in config for item in sublist]
                ]
                predicted_time = float(subprocess.check_output([target_runner, *new_args]))
                print(f'{correct_time} {predicted_time}')
                wf.write(f'{correct_time} {predicted_time}\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original_file')
    parser.add_argument('output_file')
    parser.add_argument('target_runner')
    parser.add_argument('--correct-switch', action='store_true', default=True, help='guess the correct switch by using double dash')

    args = parser.parse_args()
    main(args.original_file, args.output_file, args.target_runner, do_correct_switch=args.correct_switch)
