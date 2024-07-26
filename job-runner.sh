#/bin/bash
################################################################################
# Script: job-runner.sh
# Written By: Jason Maloney (jmmaloney@mdanderson.org)
# Date: 2022-07-27
#
# This script performs the following:
# - Validates that a kubectl configuration file exists at ~/.kube/config and is readable.
# - Performs validations on the provided job file to ensure it is a "kind: Job" file, and defines a "namespace: " value.
# - If the job file validations pass, the script will then extract the job name, namespace, and primary container name from the file.
# - Script will deploy the Job YAML to K8S and then follow the pod logs for the primary container and collect the output to a log file.
# - Script will create and execute a log-runner script.
#   - The .log-runner.sh script will ensure that there is always a "kubectl logs" processes continuously tailing the primary container log,
#   so long as the job is active.  The script will stop when it finds that the job state is 'complete' or the job is no longer running.
#   - The .log-runner.sh script regularly record the output of "kubectl describe job {jobname} -n {namespace}" to a log file.
#   - The .log-runner.sh script will record the output of "kubectl describe jobs,pods -n {namespace} -l job-name={jobname}" to a log file
#   whenever the script finds the "kubectl logs" processes has stopped and spawns a new one, so long as job state is not 'complete'.
#
# Script Requirements:
# - The job-runner.sh script requires a kubectl configuration file at ~/.kube/config and that a Job YAML file name is passed to it.
# - The provided job YAML contains only a single job definition within the file.
#
# Required External Components:
# - kubectl configuration file at ~/.kube/config
# - jobfile: Job YAML file in your current working directory
#
# Usage:  job-runner.sh <jobFile.yaml>
#
# Log Files: Logs are stored in the 'logs' directory within the current working directory that the job-runner.sh script was called from.
# - Job Runner Script Log: {jobName}-runner-{yyyy-mm-dd_hhmmss}.log
# - Container Script Log: {jobName}-{containerName}-{yyyy-mm-dd_hhmmss}.log
# - Regular Job Describe Output Log: {jobName}-describe-{yyyy-mm-dd_hhmmss}.log
# - Logger Restarted Job & Pod Describe Output Log: {jobName}-restarted-describe-{yyyy-mm-dd_hhmmss}.log
# - Job Completion Job & Pod Describe Output Log: {jobName}-completion-describe-{yyyy-mm-dd_hhmmss}.log
#
# History:
# V 1.2 Jason Maloney 2022-10-27 -- Modified script to:
# - Changed log-runner script name from log-runner.sh to .log-runner.sh.
# - Replaced 'sleep 10' after job deployment to a 'kubectl wait' command.
# - Added logic for when the kubectl logger is found not running to detect whether the job is in complete state in the log-runner.sh script.
# - Modified the describe output log file names.
# - Updated the script usage output details.
# - Added cleanup behavior to remove the .log-runner.sh script at job completion.
# - Script creates a 'done' file in the current working directory at job completion.
# - Script will delete any 'done' file in the current working directory at job-runner.sh execution.
# - Script createsa a job completion job & pod describe output Log at job completion.
# V 1.1 Jason Maloney 2022-10-07 -- Modified to allow script to be ran from a central location.
# - Script defaults to look for a kubectl config in ~/.kube/config.
# - Disabled the use of a vars.conf file and expect the user to specify jobfile as $1.
# V 1.0 Jason Maloney 2022-08-25 -- First Mature version.
# - Script has been polished enough to qualify as v1.0.
# V 0.3 Jason Maloney 2022-08-22 -- Modified script to be a generic job launcher.
# - Script will create and execute a log-runner script which runs in a loop to ensure that the container logger process is running
#   while job is active.
# V 0.2 Jason Maloney 2022-08-10 -- Modified script to be a generic job launcher.
# - Customizable variables are now read from a conf file, instead of the script being hard coded to a specific job.
# V 0.1 Jason Maloney 2022-07-27 -- Initial Development Started.
################################################################################

# Setting the base directory of the script
BASEDIR="$(pwd)"
LOGDIR="${BASEDIR}/logs"

# Create the logs dir if missing
if [ ! -d ${LOGDIR} ]
then
    mkdir -p ${LOGDIR} >/dev/null 2>&1
fi

################################################################################
# ***** [                  Define Script Usage Block                   ] ***** #
################################################################################
USAGE () {
    echo -e "\nUsage: job-runner.sh <jobFile.yaml>\n"
    echo -e "Job File: Name of your Kubernetes Job yaml file found within your current working directory.\n"
    echo -e "Log Files: Logs are stored in the 'logs' directory within the current working directory that the job-runner.sh script was called from."
    echo -e "\t- Job Runner Script Log: {jobName}-runner-{yyyy-mm-dd_hhmmss}.log"
    echo -e "\t- Container Script Log: {jobName}-{containerName}-{yyyy-mm-dd_hhmmss}.log"
    echo -e "\t- Regular Job Describe Output Log: {jobName}-describe-{yyyy-mm-dd_hhmmss}.log"
    echo -e "\t- Logger Restarted Job & Pod Describe Output Log: {jobName}-restarted-describe-{yyyy-mm-dd_hhmmss}.log"
    echo -e "\t- Job Completion Job & Pod Describe Output Log: {jobName}-completion-describe-{yyyy-mm-dd_hhmmss}.log\n"
}

################################################################################
# ***** [                      Validate Variables                      ] ***** #
################################################################################

export kubeconfig=~/.kube/config
export jobfile="$1"

    # Test that kubeconfig var is set & not empty -- exit script if it is not correct.
    # Desired state: var expected to exist & contain a value
    if ! [[ -v kubeconfig && -n ${kubeconfig} && -f ${kubeconfig} ]]
    then
        echo -e "\n$(date +'%F_%H%M%S'): No kubectl config file found at ${HOME}/.kube/config or file is empty. Exiting.\n"
        exit 1
    fi

    # Test that jobfile var is set & not empty -- exit script if it is not correct.
    # Desired state: var expected to exist & contain a value
    if  [[ -v jobfile && -n ${jobfile} ]]
    then
        # Verify that the provided file is a kind: job file by looking for 'kind: Job' string in file.
        validateJob="$(grep -c -i 'kind: Job' ${jobfile})"

        if [[ "${validateJob}" -eq "1" ]]
        then
            # Extract the namespace from the provided job file.
            validateNS="$(sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*#/d' ${jobfile} | grep 'kind: Job' --after-context=5| grep 'namespace:' | tr -d '[:space:]' | awk -F ':' '{ print $2 }'| wc -l)"

            if [[ "${validateJob}" -eq "1" ]]
            then
                # Extract the namespace from the provided job file.
                namespace="$(sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*#/d' ${jobfile} | grep 'kind: Job' --after-context=5| grep 'namespace:' | tr -d '[:space:]' | awk -F ':' '{ print $2 }')"

                # Test that namespace var is set & not empty -- exit script if it is not correct.
                # Desired state: var expected to exist & contain a value
                if ! [[ -v namespace && -n ${namespace} ]]
                then
                    echo -e "\n$(date +'%F_%H%M%S'): The namespace variable not set or is empty. Script Exiting.\n"
                    exit 1
                fi

            else
                echo -e "\n$(date +'%F_%H%M%S'): The job file ${jobfile} does not contain a Kubernetes namespace. Script Exiting.\n"
                exit 1
            fi

        else
            echo -e "\n$(date +'%F_%H%M%S'): The job file ${jobfile} is not a valid Kubernetes Job Type. Script Exiting.\n"
            exit 1
        fi

    else
        echo -e "\n$(date +'%F_%H%M%S'): No job file was provided to the script, or the provided job file is empty. Script Exiting."
        USAGE
        exit 1
    fi

################################################################################
# ***** [                   Setup Log Dir & Log File                   ] ***** #
################################################################################

# Extract the job name.
# Will see if we need to add validation that the jobName var is not empty
jobName="$(sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*#/d' ${jobfile} | grep 'kind: Job' --after-context=5 | grep -v "job-name:"| grep 'name:' | tr -d '[:space:]' | awk -F ':' '{ print $2 }')"

# Extract primary container name.
# Will see if we need to add validation that the containerName var is not empty
containerName="$(sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*#/d' ${jobfile} | grep 'containers:' --after-context=5 |grep -v "job-name:"| grep 'name:'| tr -d '[:space:]' | awk -F ':' '{ print $2 }')"

# Setting the script & container log file names now that we've extracted a container name.
LOGFILE="${LOGDIR}/${jobName}-runner-$(date +'%F_%H%M%S').log"
CONTAINERLOGFILE="${LOGDIR}/${jobName}-${containerName}-$(date +'%F_%H%M%S').log"
JOBDESCRIBELOGFILE="${LOGDIR}/${jobName}-describe-$(date +'%F_%H%M%S').log"

# Ensuring that the job isn't already running.
JobStatus="$(kubectl --kubeconfig="${kubeconfig}" -n "${namespace}" get job "${jobName}" >/dev/null 2>&1; echo $?)"

if ! [[ "${JobStatus}" -eq "1" ]]
then
    echo -e "$(date +'%F_%H%M%S'): A Job with the name ${jobName} is already deployed. Script Exiting" |tee -a $LOGFILE
    exit 1
fi

################################################################################
# ***** [                    Setup External Logging                    ] ***** #
################################################################################

# Function will write the log-runner.sh script to the basedir
WRITELOGRUNNER () {
truncate -s 0 ${BASEDIR}/.log-runner.sh >/dev/null 2>&1
cat <<EOF >> ${BASEDIR}/.log-runner.sh
#!/bin/bash
################################################################################
# Script: log-runner.sh
# Generated on $(date).

# Set Vars
LRBASEDIR="${BASEDIR}"
LRLOGDIR="${LOGDIR}"
LRKUBECONFIG="${kubeconfig}"
LRJOBNAME="${jobName}"
LRCONTAINERNAME="${containerName}"
LRNAMESPACE="${namespace}"
LRLOGFILE="${LOGFILE}"
LRCONTAINERLOGFILE="${CONTAINERLOGFILE}"
LRJOBDESCRIBELOGFILE="${JOBDESCRIBELOGFILE}"
MyUser="$(whoami)"

# Logging that the log-runner has started.
echo -e "\$(date +'%F_%H%M%S'): The log-runner.sh script for job/\${LRJOBNAME} is now running." >> \${LRLOGFILE}

# Check if Job is Running.
JobStatus="\$(kubectl --kubeconfig="\${LRKUBECONFIG}" -n "\${LRNAMESPACE}" get job "\${LRJOBNAME}" >/dev/null 2>&1; echo \$?)"

while [ "\${JobStatus}" -eq "0" ];do

    # Set the interval that the loop will recheck the log tailer status.
    sleep 30
	
    # Recheck if Job is Running
    JobStatus="\$(kubectl --kubeconfig="\${LRKUBECONFIG}" -n "\${LRNAMESPACE}" get job "\${LRJOBNAME}" >/dev/null 2>&1; echo \$?)"

    if [[ "\${JobStatus}" -gt "0" ]]
    then
        # Logging that the job is no longer running.
        echo -e "\$(date +'%F_%H%M%S'): The job/\${LRJOBNAME} is no longer running. Log-runner.sh script is stopping." >> \${LRLOGFILE}
        touch \${LRBASEDIR}/done
        rm -f \${LRBASEDIR}/.log-runner.sh > /dev/null 2>&1
		exit 0
    fi

    # Dumping the description of the job being monitored.
    \$(kubectl --kubeconfig="\${LRKUBECONFIG}" -n "\${LRNAMESPACE}" describe job "\${LRJOBNAME}" >\${LRJOBDESCRIBELOGFILE} 2>&1)

    # Check if there is already a process logging the container log.
    # This checks for a kubectl logs process that matches the job & namespace is actively running.
    # Could look to enhance the accuracy of the script by comparing the returned PID against the status of any associated PID with the open container log file. Keeping simple for now.
    loggerStatus="\$(ps -f -u \${MyUser} | grep "kubectl" | grep ""\${LRNAMESPACE}" "job/\${LRJOBNAME}"" | grep -v "grep" >/dev/null 2>&1; echo \$?)"

    if ! [[ "\${loggerStatus}" -ge "1" ]]
    then
        # Kubectl log tailer process found
        loggerCount="\$(ps -f -u \${MyUser} | grep "kubectl" | grep "\$LRNAMESPACE job/\$LRJOBNAME" |grep -v 'grep' | wc -l)"

        if [[ "\${loggerCount}" -eq "1" ]]
        then
            # Touching the job-runner.sh log file to update the timestamp so you can tell the log-runner.sh script hasn't hung.
            \$(touch \${LRLOGFILE})
        fi

    else
        echo -e "\$(date +'%F_%H%M%S'): Kubectl Log tailer for job/\${LRJOBNAME} appears to no longer be running." >> \${LRLOGFILE}
        echo -e "\$(date +'%F_%H%M%S'): Checking the current state of job/\${LRJOBNAME}." >> \${LRLOGFILE}

        # Get State of job:
        # JobStateCheck1 checks if the job is currently 'active' (equates to running / not complete).
        # JobStateCheck2 checks if the job is currently 'complete'.
        JobStateCheck1="\$(kubectl --kubeconfig="\${LRKUBECONFIG}" -n "\${LRNAMESPACE}" get job "\${LRJOBNAME}" -o=jsonpath='{.status}'|grep '"active":1' >/dev/null 2>&1; echo \$? )"
        JobStateCheck2="\$(kubectl --kubeconfig="\${LRKUBECONFIG}" -n "\${LRNAMESPACE}" get job "\${LRJOBNAME}" -o=jsonpath='{.status}'|grep '"status":"True","type":"Complete"' >/dev/null 2>&1; echo \$?)"

        if [ "\${JobStateCheck1}" -eq "0" -a "\${JobStateCheck2}" -eq "1" ]
        then
            echo -e "\$(date +'%F_%H%M%S'): Job/\${LRJOBNAME} appears to still be active.  Starting the logger again." >> \${LRLOGFILE}
            nohup kubectl --kubeconfig="\${LRKUBECONFIG}" logs -n \${LRNAMESPACE} job/\${LRJOBNAME} -f --all-containers --pod-running-timeout=30s --prefix --timestamps --ignore-errors=true >> \${LRCONTAINERLOGFILE} 2>&1 &

            # Dumping the description of the job & pods being monitored.
            # Goal here is to try and capture all of the current pods (starting, running, terminating, error, ect) at this time.
            \$(kubectl --kubeconfig="\${LRKUBECONFIG}" -n "\${LRNAMESPACE}" describe jobs,pods -l "job-name=\${LRJOBNAME}" >>"\${LRLOGDIR}/\${LRJOBNAME}-restarted-describe-\$(date +'%F_%H%M%S').log" 2>&1)

        else
            if [ "\${JobStateCheck1}" -eq "1" -a "\${JobStateCheck2}" -eq "0" ]
            then
                echo -e "\$(date +'%F_%H%M%S'): Job/\${LRJOBNAME} appears to have completed.  Log-runner.sh script is stopping." >> \${LRLOGFILE}
                # Dumping the description of the job & pods being monitored.
                # Goal here is to try and capture all of the current pods (starting, running, terminating, error, ect) at this time.
                \$(kubectl --kubeconfig="\${LRKUBECONFIG}" -n "\${LRNAMESPACE}" describe jobs,pods -l "job-name=\${LRJOBNAME}" >>"\${LRLOGDIR}/\${LRJOBNAME}-completion-describe-\$(date +'%F_%H%M%S').log" 2>&1)	
                touch \${LRBASEDIR}/done
                rm -f \${LRBASEDIR}/.log-runner.sh > /dev/null 2>&1
                JobStatus="1"
            fi	
        fi
    fi
done
EOF
# Sets the exec permission on the generated log-runner.sh script.
chmod +x ${BASEDIR}/.log-runner.sh >/dev/null 2>&1
}

################################################################################
# ***** [                    Execute Jobs & Scripts                    ] ***** #
################################################################################

# Removing any existing done files in current working directory
DONEFILE="${BASEDIR}/done"
if [[ -f "${DONEFILE}" ]]; then
    rm -f ${DONEFILE} >/dev/null 2>&1
fi

# Deploying the job
echo -e "$(date +'%F_%H%M%S'): Deploying the ${jobfile} job file." |tee -a $LOGFILE
kubectl --kubeconfig="${kubeconfig}" create -f ${jobfile}|tee -a $LOGFILE

# Waiting for the job pod to become 'Ready'
echo -e "$(date +'%F_%H%M%S'): Waiting for the pod of job/${jobName} to be ready before starting the log tailing process." |tee -a $LOGFILE
kubectl --kubeconfig="${kubeconfig}" wait -n ${namespace} --for=condition=Ready pod -l "job-name=${jobName}" |tee -a $LOGFILE

#nohup kubectl --kubeconfig="${kubeconfig}" logs -n ${namespace} -l job-name=${jobName} -c ${containerName} --timestamps -f > $CONTAINERLOGFILE 2>&1 &
echo -e "$(date +'%F_%H%M%S'): Starting to tail the container log." |tee -a $LOGFILE
nohup kubectl --kubeconfig="${kubeconfig}" logs -n ${namespace} job/${jobName} -f --all-containers --pod-running-timeout=30s --prefix --timestamps --ignore-errors=true > $CONTAINERLOGFILE 2>&1 &

# Calling the WRITELOGRUNNER function to generate the log-runner.sh script.
echo -e "$(date +'%F_%H%M%S'): Creating the log-runner.sh script." |tee -a $LOGFILE
WRITELOGRUNNER

# Running the log-runner.sh script
echo -e "$(date +'%F_%H%M%S'): Executing the log-runner.sh script." |tee -a $LOGFILE
nohup ${BASEDIR}/.log-runner.sh >> $LOGFILE 2>&1 &