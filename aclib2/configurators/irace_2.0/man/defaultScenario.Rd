\name{defaultScenario}
\alias{defaultScenario}
\title{defaultScenario}
\description{
  'defaultScenario' returns the default scenario settings of
  \pkg{irace}.
}
\usage{
 defaultScenario(scenario = list())
}
\arguments{
   \item{scenario}{A list where tagged elements correspond to scenario
    settings of \pkg{irace}.}
}
%
\value{
  A list indexed by the \pkg{irace} parameter names,
  containing the default values for each parameter, except for those
  already present in the scenario passed as argument.
 The scenario list contains the following elements: 

  General:
  \itemize{
    \item scenarioFile: Path of the file that describes the configuration scenario setup and other irace settings. (Default: "./scenario.txt")
    \item debugLevel: Debug level of the output of \code{irace}. Set this to 0 to silence all debug messages. Higher values provide more verbose debug messages. (Default: 0)
    \item seed: Seed of the random number generator (must be a positive integer, NA means use a random seed). (Default: \code{NA})
    \item execDir: Directory where the experiments will be run. (Default: "./")
    \item logFile: Path of the file to save tuning results as an R dataset, either absolute path or relative to execDir. (Default: "./irace.Rdata")    
  }
  Elitist irace:
  \itemize{
     \item elitist: Enable/disable elitist irace. (Default: 1)
     \item elitistNewInstances: Number of instances added to the execution list before previous instances in elitist irace. (Default: 1)
     \item elitistLimit: Limit for the elitist race, number statistical test without elimination peformed. Use 0 for no limit. (Default: 2)
  }
  \code{irace} internal: For most of these parameters is adviced to use the default settings.
  \itemize{
     \item sampleInstances: Enable/disable the sampling of instances. (Default: 1)
     \item nbIterations: Number of iterations to be performed by \code{irace}.
     \item nbExperimentsPerIteration: Number of experiments (runs) per iteration. 
     \item nbConfigurations: Number of configurations to be sampled and evaluated at each iteration.
     \item mu: Parameter used to define the number of configurations sampled and evaluated at each iteration. (Default: 5)
     \item minNbSurvival: Minimum number of configurations needed to continue executing a race. 
     \item softRestart: Enable/disable the soft restart strategy that avoids premature convergence of the probabilistic model. (Default: 1)
     \item softRestartThreshold: Soft restart threshold value for numerical parameters. If \code{NA}, it computed as \code{10^-digits}. (Default: \code{NA})
  }
  Target algorithm parameters:
  \itemize{
     \item parameterFile: Path to the file that contains the description of the parameters to be tuned. See the template. (Default: "./parameters.txt")
     \item digits: Indicates the number of decimal places to be considered for the real parameters. (Default: 4)
     \item forbiddenExps: MANUEL
     \item forbiddenFile: Path to a file that contains a list of logical expressions that cannot be \code{TRUE} for any evaluated configuration. If empty or \code{NULL}, do not use forbidden expressions. (Default: "")
  }
  Target algorithm execution:
  \itemize{
     \item targetRunner: Path to the script called for each configuration that launches the algorithm to be tuned. See templates. (Default: "./target-runner")
     \item targetRunnerRetries: Number of times to retry a call to \code{targetRunner} if the call failed. (Default: 0)
     \item targetRunnerData: Optional data passed to \code{targetRunner}. This is ignored by the default \code{targetRunner} function, but it may be used by custom \code{targetRunner} functions to pass persistent data around.
     \item targetRunnerParallel: Optional R function to provide custom parallelization of \code{targetRunner}.
     \item targetEvaluator: Path to the script that provides a numeric value for each configuration. See templates. (Default: "")
     \item deterministic: If the target algorithm is deterministic, configurations will be evaluated only once per instance. (Default: 0)
     \item parallel: Number of calls to \code{targetRunner} to execute in parallel. 0 or 1 mean disabled. (Default: 0) 
     \item loadBalancing: Enable/disable load-balancing when executing experiments in parallel. Load-balancing makes better use of computing resources, but increases communication overhead. If this overhead is large, disabling load-balancing may be faster. (Default: 1)
     \item mpi: Enable/disable MPI. Use Rmpi to execute targetRunner in parallel (parameter parallel is the number of slaves). (Default 0)
     \item sgeCluster: Enable/disable SGE cluster mode. Use qstat to wait for cluster jobs to finish (targetRunner must invoke qsub). (Default: 0)
  }
  Initial configurations:
  \itemize{
      \item configurationsFile: Path to a file that contains a set of initial configurations. If empty or \code{NULL} do not use a initial configurations. (Default: "")
  }
  Training instances:
  \itemize{
     \item instances: Array of the instances to be used in the \code{targetRunner}.
     \item instancesList: Data frame that constains the instance and seeds used in the tuning. The order of this data frame indicate the order in which the instances have been used.
        \itemize{
          \item instance: The instance index (corresponding to the \code{instance} array).
          \item seed: Seed to be used with the corresponding instance.
        }
     \item instances.extra.params: Array that contains the extra parameters defined per instance, the name of each element corresponds to the instance name in the \code{instances} array. When no extra parameters are provided this element is \code{NULL}.
     \item trainInstancesDir: Directory where tuning instances are located; either absolute path or relative to current directory.. If no \code{trainInstancesFiles} is provided all the files in \code{trainInstancesDir} will be listed as instances. (Default: "./Instances")
     \item trainInstancesFile: Path to a file that contains a list of instances and optionally additional parameters for them. If \code{trainInstancesDir} is provided \code{irace} will search for the files in this folder. (Default: "")
  }
  Tuning budget:
  \itemize{
     \item maxExperiments: Maximum number of runs (invocations of \code{targetRunner}) that will be performed. It determines the maximum budget of experiments for the tuning. (Default: 0)
     \item maxTime: Maximum total execution time for the executions of \code{targetRunner}. targetRunner must return two values: quality time. (Default: 0)
     \item budgetEstimation: Percentage of the time budget used to estimate the mean computation time of a configuration. Only used when maxTime is provided. (Default: 0.02)
  } 
  Statistical test:
  \itemize{
     \item testType: Statistical test used for elimination. (Default: "F-test") 
     \item confidence: Confidence level of the statistical test. (Default: 0.95)
     \item firstTest: Number of instances evaluated before the first elimination test. (Default: 5)
     \item eachTest: Number of instances evaluated between elimination tests. (Default: 1) 
  }
  Recovery:
  \itemize{
     \item recoveryFile: Path to an irace log file used to recover the \code{irace} execution. (Default: "") 
  }
  Testing: 
  \itemize{
     \item testNbElites: Number of elite configurations returned by
  irace to be tested if test instances are provided. (Default: 1)
     \item testIterationElites: Enable/disable testing the elite configurations found at each iteration. (Default: 0)
     \item testInstancesDir: Directory where testing instances are located, either absolute or relative to current directory. (Default: "")
     \item testInstancesFile: Path to a file containing a list of test instances and optionally additional parameters for them. (Default: "")
     \item testInstances: Array of the instances to be used in the \code{targetRunner} when executing the testing.
     \item testInstances.extra.params: Array that contains the extra parameters defined per instance, the name of each element corresponds to the instance name in the \code{testInstances} array. When no extra parameters are provided this element is \code{NULL}.
  }
}
\seealso{
  \code{\link{readScenario}} for reading a configuration scenario from a
  file.
  \code{\link{checkScenario}} to check that the scenario is valid.
  \code{\link{printScenario}} prints the given scenario.
}
\author{Manuel López-Ibáñez and Jérémie Dubois-Lacoste}
