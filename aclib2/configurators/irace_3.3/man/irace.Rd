% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/irace.R
\name{irace}
\alias{irace}
\title{irace}
\usage{
irace(scenario, parameters)
}
\arguments{
\item{scenario}{Data structure containing \pkg{irace} settings.The data structure 
has to be the one returned by the function \code{\link{defaultScenario}} and 
\code{\link{readScenario}}.}

\item{parameters}{Data structure containing the parameter definition. The data 
structure has to be the one returned by the function \code{\link{readParameters}}.}
}
\value{
A data frame with the set of best algorithm configurations found by \pkg{irace}. 
 The data frame has the following columns:
 \describe{
   \item{\code{.ID.}}{Internal id of the candidate configuration.}
   \item{\code{Parameter names}}{One column per parameter name in \code{parameters}.}
   \item{\code{.PARENT.}}{Internal id of the parent candidate configuration.}
 }
 Additinally, this function saves an R data file containing an object called
 \code{iraceResults}. The path of the file is indicated in \code{scenario$logFile}. 
 The \code{iraceResults} object is a list with the following structure:
 
\describe{

    \item{\code{scenario}}{The scenario R object containing the \pkg{irace}
    options used for the execution. See \code{\link{defaultScenario}} help
    for more information.}
 
    \item{\code{parameters}}{The parameters R object containing the
    description of the target algorithm parameters. See
    \code{\link{readParameters}}.}
 
    \item{\code{allConfigurations}}{The target algorithm configurations
    generated by \pkg{irace}. This object is a data frame, each row is a
    candidate configuration, the first column (\code{.ID.}) indicates the
    internal identifier of the configuration, the following columns
    correspond to the parameter values, each column named as the parameter
    name specified in the parameter object. The final column
    (\code{.PARENT.})  is the identifier of the configuration from which
    model the actual configuration was sampled.}
  
    \item{\code{allElites}}{A list that contains one element per iteration,
    each element contains the internal identifier of the elite candidate
    configurations of the corresponding iteration (identifiers correspond to
    \code{allConfigurations$.ID.}).}
    
    \item{\code{iterationElites}}{A vector containing the best candidate
    configuration internal identifier of each iteration. The best
    configuration found corresponds to the last one of this vector.}
    
    \item{\code{experiments}}{A matrix with configurations as columns and
    instances as rows. Column names correspond to the internal identifier of
    the configuration (\code{allConfigurations$.ID.}).}
    
    \item{\code{experimentLog}}{A matrix with columns \code{iteration,
    instance, configuration, time}.  This matrix contains the log of all the
    experiments that \pkg{irace} performs during its execution.  The
    instance column refers to the index of the \code{scenario$instancesList}
    data frame. Time is saved ONLY when reported by the targetRunner.}

    \item{\code{softRestart}}{A logical vector that indicates if a soft
    restart was performed on each iteration. If \code{FALSE}, then no soft
    restart was performed.}

    \item{\code{state}}{A list that contains the state of \pkg{irace}, the
    recovery is done using the information contained in this object.}
    
    \item{\code{testing}}{A list that constains the testing results. The
    elements of this list are: \code{experiments} a matrix with the testing
    expriments of the selected configurations in the same format as the
    explained above and \code{seeds} a vector with the seeds used to execute
    each experiment.}

}
}
\description{
\code{irace} implements iterated Race. It receives some parameters to be tuned 
  and returns the best configurations found, namely, the elite configurations 
  obtained from the last iterations (and sorted by rank).
}
\details{
The function \code{irace} executes the tuning procedure using 
 the information provided in \code{scenario} and \code{parameters}. Initially it checks 
 the correctness of \code{scenario} and recovers a previous execution if 
 \code{scenario$recoveryFile} is set. A R data file log of the execution is created 
 in \code{scenario$logFile}.
}
\examples{
\dontrun{
parameters <- readParameters("parameters.txt")
scenario <- readScenario(filename = "scenario.txt",
                         scenario = defaultScenario())
irace(scenario = scenario, parameters = parameters)
}

}
\seealso{
\describe{
 \item{\code{\link{irace.main}}}{a higher-level command-line interface to \code{irace}.}
 \item{\code{\link{readScenario}}}{for reading a configuration scenario from a file.}
 \item{\code{\link{readParameters}}}{read the target algorithm parameters from a file.}
 \item{\code{\link{defaultScenario}}}{returns the default scenario settings of \pkg{irace}.}
 \item{\code{\link{checkScenario}}}{to check that the scenario is valid.}
}
}
\author{
Manuel López-Ibáñez and Jérémie Dubois-Lacoste
}
