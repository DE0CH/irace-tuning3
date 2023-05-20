#TODO: not very sure how this works, don't use it yet
raise NotImplementedError('This is not implemented yet.')

import epm.webserver.flask_worker
import sys

if __name__ == '__main__':
    cost, _ = epm.webserver.flask_worker.send_procedure([sys.argv[1:]]) #TODO: check if it is actually from 1
    print(cost[0]) #TODO: this maybe a numpy array, so need to check if this actually gets a single number.
    print(cost[0]) 
    