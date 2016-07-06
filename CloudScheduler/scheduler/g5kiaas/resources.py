#!/usr/bin/env python
"""
@author: Tran Ngoc Minh
"""
import subprocess, threading
import os, sys, traceback, time
from globals import Environment

from remote.connection import RemoteConn

class ResourceProvision:
    def __init__(self):
        self.conn = RemoteConn()
        self.MACHINES = {}
        self.jobs = []
        self.reserve = []

        try:
            lines = open(os.path.join(Environment.CONF, "resource.conf"), 
                         "r").readlines()
            lines = filter(lambda line: not(line.startswith("#") or
                                line.startswith("//")), lines)
            
            Environment.WALLTIME = int(filter(lambda line: "WALLTIME" in 
                                line, lines)[0].split("=")[1].strip())

            for i in range(len(lines)):
                if "RESOURCES" in lines[i]:
                    lines = lines[i + 1:]    
                    break                                                                                     
            lines = filter(lambda line: not("WALLTIME" in line), lines)
        except:
            print traceback.print_exc()
            sys.exit(1)

        deploy = True
        for line in lines:
            if line.startswith("JOBS"):
                deploy = False
                self.jobs.extend(map(lambda job: {"Site" : job.split("-")[0],
                        "Cluster" : job.split("-")[1], "jobID" : 
                        int(job.split("-")[2])},line.split(":")[1].split()))
            else:    
                site, cluster, num = line.split("\t")
                self.reserve.append({"Site" : site.strip(), "Cluster" : 
                                     cluster.strip(), "Num" : int(num.strip())})
        
        try:    
            self.__get_jobs()
            self.__get_machines()    
        except:
            print traceback.print_exc()
            print "Failed making reservations or retrieving machines."
            return
        
        if deploy:
            self.__deploy_env()
        
    def __get_jobs(self):
        if self.jobs == []:
            for reservation in self.reserve:
                site = reservation["Site"]
                cluster = reservation["Cluster"]                
                cmd = """oarsub "sleep 86400" -l slash_22=1+{"cluster='%s'"}nodes=%d,walltime=%d -t deploy""" \
                        % (cluster, reservation["Num"], Environment.WALLTIME)                
                output = self.conn.run(site, Environment.USER, cmd)
                
                job_id = filter(lambda line: line.startswith("OAR_JOB_ID"), 
                                output.split())[0]
                job_id = int(job_id.split('=')[1].strip())
                
                self.jobs.append({"Site" : site, "Cluster" : cluster, 
                                  "jobID" : job_id})
                time.sleep(5)

            print "Waiting for 30sec before requesting machine data..."
            time.sleep(30)
                    
    def __get_machines(self):
        for job in self.jobs:
            site = job["Site"]
            if not self.MACHINES.has_key(site):
                self.MACHINES[site] = {}
            
            cluster = job["Cluster"]
            cmd = """oarstat -j %d -p | oarprint host -P %s,%s,%s,%s -F "%% %% %% %%" -f -""" \
                    % (job["jobID"], Environment.ATTR["IP"], Environment.ATTR["HOSTNAME"], \
                       Environment.ATTR["RAM"], Environment.ATTR["Frq"])
                    
            output = self.conn.run(site, Environment.USER, cmd)
            mach = output.split("\r\n")
            mach = filter(lambda line: line.strip() != "", mach)
            mach = mach[1:]

            self.MACHINES[site][cluster] = {"Machines" : [], "Characteristics" : {}, \
                                            "TotalNum" : 0, "AvailableMachineNum" : 0}
            ram = 0
            frq = 0
            for m in mach:
                ip, host, ram, frq = m.split()
                self.MACHINES[site][cluster]["Machines"].append({"ip" : ip, "hostname" : host})
                        
            cmd = """oarstat -j %d -p | oarprint core -P host -F "%% " -f -""" % (job["jobID"],)
            output = self.conn.run(site, Environment.USER, cmd)
            
            num_cores = output.split("\r\n")
            num_cores = filter(lambda line: line.strip() != "", num_cores)
            num_cores = num_cores[1:]
            
            self.MACHINES[site][cluster]["TotalNum"] = len(mach)
            self.MACHINES[site][cluster]["AvailableMachineNum"] = len(mach)
            self.MACHINES[site][cluster]["Characteristics"] = { "Cores" : len(num_cores)/len(mach), \
                                                               "RAM" : ram, "Frq" : frq}
            self.MACHINES[site][cluster]["JobID"] = job["jobID"]
           
    def __deploy_env(self):
        procs = []
        for site  in self.MACHINES:
            m = []
            for c in self.MACHINES[site].values():
                m.extend(map(lambda mac:mac["hostname"], c["Machines"]))
                   
            procs.append(threading.Thread(target=self.__remote_deploy, args = (site,m)))
            procs[-1].start()

        print "Waiting for deployment..."    
        for t in procs:
            t.join()        
        print "Environment deployed on all machines"        
        
    def __remote_deploy(self, site, machine_list):
        try:
            cmd = """rm /home/%s/machine.list""" % Environment.USER
            self.conn.run(site, Environment.USER, cmd)
        except:
            print "Machine file does not exist!!"
        
        cmd = """touch /home/%s/machine.list""" % Environment.USER
        self.conn.run(site, Environment.USER, cmd)
        
        cmd = """cat > /home/%s/machine.list <<EOF\n%s\nEOF""" % \
                    (Environment.USER, "\n".join(machine_list))
        self.conn.run(site, Environment.USER, cmd)
        
        print time.strftime("START : %Y-%m-%d %H:%M:%S", time.gmtime())
        cmd = """kadeploy3 -f /home/%s/machine.list -e wheezy-x64-big -k /home/%s/.ssh/id_rsa.pub""" \
                 %(Environment.USER, Environment.USER)
        
        self.conn.run(site, Environment.USER, cmd)
        print time.strftime("DONE  : %Y-%m-%d %H:%M:%S", time.gmtime())
        print "Done deploying machines from site %s " % site
    
    def __start_mond(self):
        for site  in self.MACHINES:       
            for c in self.MACHINES[site].values():
                for ip in map(lambda mac:mac["ip"], c["Machines"]):
                    cmd = """scp %s root@%s:/root""" % (Environment.MOND_PATH, ip)
                    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).wait()
                    subprocess.Popen("ssh %s chmod +x /root/mond" % ip, shell=True, 
                                     stdout=subprocess.PIPE).wait()

                    cmd = "nohup python /root/mond &"
                    subprocess.Popen("ssh %s %s" % (ip, cmd), shell=True, 
                                     stdout=subprocess.PIPE).wait()
                    
class ResourceSet:
    def __init__(self, site, cluster, resources):
        """
        {'grenoble': {'genepi': {'AvailableMachineNum': 3,
                         'Characteristics': {'Cores': 8,
                                             'Frq': '2.5',
                                             'RAM': '8192'},
                         'JobID': 1476039,
                         'Machines': [{'hostname': 'genepi-31.grenoble.grid5000.fr',
                                       'ip': '172.16.16.31'},
                                      {'hostname': 'genepi-4.grenoble.grid5000.fr',
                                       'ip': '172.16.16.4'},
                                      {'hostname': 'genepi-32.grenoble.grid5000.fr',
                                       'ip': '172.16.16.32'}],
                         'TotalNum': 3}}}
        """
        
        self.Path = "/%s/%s" % (site, cluster)
        self.site = site
        self.cluster = cluster
        self.JobID = resources["JobID"]
        self.attributes = resources["Characteristics"]
        self.devices = {}
        
        for m in resources["Machines"]:
            self.devices[m["ip"]] = {"Available" : resources["Characteristics"].copy(), 
                                     "Hostname" : m["hostname"], 
                                     "Utilization" : {"CPU" : 0, "RAM" : 0, }}
            
