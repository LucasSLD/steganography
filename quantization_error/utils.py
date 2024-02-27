import sys, os, shutil, traceback, subprocess
import numpy as np
import pickle
# from .unholyException import UnholyException
# from . import global_vars
from multiprocessing import Pool, cpu_count
from fnmatch import fnmatch
from glob import glob
from math import ceil

os.environ["OMP_NUM_THREADS"] = "1"

def pool_func(map_args):
    try:
        map_args[0].__name__ # map_args[0] is cls
    except AttributeError:
        raise UnholyException("First argument must be a class object")

    try:
        map_args[0].run # cls.run()
    except AttributeError:
        raise UnholyException("Could not call " + map_args[0].__name__ + ".run()")

    try:
        return map_args[0].run(*map_args[1:])
    except Exception as e:
        Utils.output_fatal(str(sys.exc_info()[1]))
        print(10*'-')
        print(Utils.get_traceback())
        print(10*'-')
        raise e

class Utils():
    """ Implement holy utility methods
    """

    GLOBAL_ERROR_FILE = 'error'
    GLOBAL_RESULT_FILE = 'result'
    GLOBAL_NBFILES_FILE = 'nbfiles'
    GLOBAL_PROGRESS_FILE = 'progress'

    ###########################################################################
    # Multithread methods

    @staticmethod
    def get_cpu_count():
        return cpu_count()

    @staticmethod
    def multithread(args):
        """
        Runs "args" as list on several CPUs.
        First element of "args" must be a class object, that contains a function "run()".
        All other elements are function run() arguments
        """
        Utils.output_step("Starting multithreading")
        pool = Pool(min(cpu_count(), global_vars.global_nbcore))

        ret = None
        try:
            ret = pool.map(pool_func, args)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.close()
            pool.terminate()
            pool.join()
        return ret

    @staticmethod
    def send_progress():
        with open(global_vars.global_comdir + '/' + Utils.GLOBAL_PROGRESS_FILE, "a") as f:
            f.write('a')

    @staticmethod
    def send_nb_files(NB_FILES):
        with open(global_vars.global_comdir + '/' + Utils.GLOBAL_NBFILES_FILE, "w") as f:
            f.write(str(NB_FILES))

    @staticmethod
    def send_interrupt():
        if not Utils.exists(global_vars.global_comdir + '/' + Utils.GLOBAL_ERROR_FILE):
            with open(global_vars.global_comdir + '/' + Utils.GLOBAL_ERROR_FILE, "w") as f:
                f.write('error')


    ###########################################################################
    # Display methods

    @staticmethod
    def output_title(string):
        print(80*'-')
        print((len(string) + 8)*'*')
        print("*** " + string + " ***")
        print((len(string) + 8)*'*')
        sys.stdout.flush()

    @staticmethod
    def output_debug(string):
        print("Debug: " + string)
        sys.stdout.flush()

    @staticmethod
    def output_step(string):
        print("[" + string + "]")
        sys.stdout.flush()

    @staticmethod
    def output_log(string):
        print("LOG: " + string)
        sys.stdout.flush()

    @staticmethod
    def output_info(string):
        print("\033[34m{}\033[00m".format("INFO: " + string))
        sys.stdout.flush()

    @staticmethod
    def output_warning(string):
        print("\033[33m{}\033[00m".format("WARNING: " + string))
        sys.stdout.flush()

    @staticmethod
    def output_error(string):
        print("\033[31m{}\033[00m".format("ERROR: " + string))
        sys.stdout.flush()

    @staticmethod
    def output_fatal(string):
        print("\033[31m{}\033[00m".format("FATAL: " + string))
        sys.stdout.flush()

    @staticmethod
    def output_result(string):
        print("*** RESULT ***")
        print("\033[34m{}\033[00m".format(string))
        print("**************")
        sys.stdout.flush()
        with open(global_vars.global_comdir + '/' + Utils.GLOBAL_RESULT_FILE, "a") as f:
            f.write(string)

    ###########################################################################
    # System methods

    @staticmethod
    def execute(cmd):
        try:
            return subprocess.call([cmd], stdin=None, shell=True)
        except Exception as e:
            Utils.output_fatal(cmd)
            Utils.output_fatal(str(sys.exc_info()[1]))
            return None

    @staticmethod
    def basename(name):
        return os.path.basename(name)

    @staticmethod
    def dirname(name):
        return os.path.dirname(name)

    @staticmethod
    def join_bag_and_img(bag, img):
        return "%06d_%s" % (bag, img)

    @staticmethod
    def split_bag_and_img(name):
        return name.split("_")

    @staticmethod
    def copy(from_, to):
        shutil.copy(from_, to)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def get_recursive_file_number(dir_path):
        files_nb = 0
        for _, _, files in os.walk(dir_path):
            files_nb += len(files)
        return files_nb

    @staticmethod
    def get_traceback():
        return traceback.format_exc()

    @staticmethod
    def getsize(file_path):
        return os.path.getsize(file_path)

    @staticmethod
    def isabs(path):
        return os.path.isabs(path)

    @staticmethod
    def isfile(path):
        return os.path.isfile(path)

    @staticmethod
    def isdir(path):
        return os.path.isdir(path)

    @staticmethod
    def listdir(dir_path):
        return os.listdir(dir_path)

    @staticmethod
    def makedirs(dir_path):
        if not Utils.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def mkdir(dir_path):
        if not Utils.exists(dir_path):
            os.mkdir(dir_path)

    @staticmethod
    def move(from_, to):
        shutil.move(from_, to)

    @staticmethod
    def remove(path):
        if Utils.exists(path):
            os.remove(path)

    @staticmethod
    def rmtree(dir_path):
        if Utils.exists(dir_path):
            shutil.rmtree(dir_path)

    ###########################################################################
    # Files methods

    @staticmethod
    def cut(files_list, num_parts):
        """
        Cuts a list of files into 'num_parts' lists of files.
        """
        files_list.sort()  # sorts the list so that the cut is unique
        count = len(files_list)
        ret = []
        for i in range(num_parts):
            part_i = files_list[i*count // num_parts: (i+1)*count // num_parts]
            ret.append(part_i)
        return ret

    @staticmethod
    def cut_by_number(files_list, num_im):
        """
        Cuts a list of files into k lists of files of 'num_im' images.
        """
        files_list.sort()  # sorts the list so that the cut is unique
        count = len(files_list)
        num_parts = np.int(np.ceil(np.float(count)/num_im))
        ret = []
        for i in range(num_parts):
            part_i = files_list[i*num_im : (i+1)*num_im]
            ret.append(part_i)
        return ret

    @staticmethod
    def cut_by_ratio(files_list, ratio):
        """
        Cuts a list of files into 2 lists of ratio and (1-ratio) elements
        """
        files_list.sort()  # sorts the list so that the cut is unique
        return (files_list[:ceil(ratio * len(files_list))], files_list[ceil(ratio * len(files_list)):])

    @staticmethod
    def cut_by_batch(files_list, num_parts):
        """
        Cuts a list of files into 'num_parts' lists of batchs.
        """
        # Separate batch and file name
        bags = []
        images = []
        for index in range(len(files_list)):
            [b, i] = Utils.split_bag_and_img(Utils.basename(files_list[index]))
            bags.append(b)
            images.append(i)
        sorted_bags = sorted(zip(bags, images))
        unique_bags = set([b for b,_ in sorted_bags])
        nb_of_bags = len(unique_bags)
        imgs_per_bag = len(files_list) // nb_of_bags

        # Create bags according to bag name
        dir_name = Utils.dirname(files_list[0])
        all_bags = [[] for b_index in range(nb_of_bags)]
        for i_index in range(len(files_list)):
            all_bags[int(bags[i_index])].append(dir_name + "/" + Utils.join_bag_and_img(int(bags[i_index]), images[i_index]))

        # Distribute bags among nodes
        res = []
        for i in range(num_parts):
            part_i = all_bags[(i*nb_of_bags // num_parts): ((i+1)*nb_of_bags // num_parts)]
            res.append(part_i)

        return res

    @staticmethod
    def find_files(directory, pattern='*'):
        """
        Recursively returns every file satisfying 'pattern'
        """
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename
            for dir in dirs:
                for f in find_files(dir, pattern):
                    yield f

    @staticmethod
    def glob(pattern):
        return glob(pattern)

    @staticmethod
    def pickle_save(file_path, obj):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def pickle_load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def pickle_dump(file_path, obj):
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def np_save(file_path, obj):
        np.save(file_path, obj)

    @staticmethod
    def np_load(file_path):
        return np.load(file_path)


    ###########################################################################
    # Util methods

    @staticmethod
    def dynamic_getter(mod_path, cls):
        """ Imports a class or function cls from a module path mod_path
        """
        mod_ = __import__(mod_path, fromlist=[cls])
        return getattr(mod_, cls)

    @staticmethod
    def get_func_arg_names(func, offset=0):
        if offset > func.__code__.co_argcount:
            return []
        return func.__code__.co_varnames[offset:func.__code__.co_argcount]

    @staticmethod
    def is_multicanal():
        if not isinstance(global_vars.global_multicanal, bool):
            raise UnholyException("Multicanal is not set (boolean). Please modify the configuration file")
        return global_vars.global_multicanal

    @staticmethod
    def get_quality():
        if not isinstance(global_vars.global_quality, int) and not isinstance(global_vars.global_quality, float):
            raise UnholyException("Quality is not set (int or float). Please modify the configuration file")
        return global_vars.global_quality

    ###########################################################################
    # Random methods

    @staticmethod
    def randint(max_=10000):
        if global_vars.global_reproductible:
            return 1
        return np.random.randint(max_)

    @staticmethod
    def randseed(seed=None):
        if seed is None:
            try:
                if global_vars.global_reproductible:
                    seed = 0
                else:
                    seed = np.fromstring(os.urandom(32),dtype=np.uint32)[0]
            except:
                seed = 0
        np.random.seed(seed)

    @staticmethod
    def randshuffle(tuple_):
        return np.random.shuffle(tuple_)

    @staticmethod
    def randsample(tuple_):
        return np.random.random_sample(tuple_)

    @staticmethod
    def randperm(max_):
        return np.random.permutation(max_)

    @staticmethod
    def urandom(size):
        return os.urandom(size)
