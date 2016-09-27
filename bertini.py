from __future__ import print_function

import collections
import math
import os
import shutil
import tempfile

def map(f, x):
    return [f(y) for y in x]

class Point(object):
    def __init__(self, coordinates):
        self._coordinates = list(coordinates)

    def __str__(self):
        return str(self.coordinates)

    def __repr__(self):
        return self.__str__()

    def eq(self, other, tol):
        n = len(self.coordinates)
        try:
            norm = math.sqrt(sum([abs(self.coordinates[i]-other.coordinates[i]) for i in range(n)]))
        except (AttributeError, IndexError, TypeError) as e: # has no attribute "coordinates", sizes are wrong, types are wrong respectively
            return False

        return norm <= tol

    def __len__(self):
        return len(self.coordinates)

    def __get_item__(self, key):
        return self.coordinates[key]

    def __set_item__(self, key, value):
        self.coordinates[key] = complex(value)

    def __contains__(self, item):
        return self.coordinates.__contains__(item)

    @property
    def coordinates(self):
        return self._coordinates

class IrreducibleComponent(object):
    def __init__(self, witness_points, dimension, component_id, dirname):
        self._witness_points = witness_points
        self._degree = len(witness_points)
        self._dimension = dimension
        self._component_id = component_id
        self._dirname = dirname

    def __str__(self):
        return "dim {0} deg {1}".format(self.dimension, self.degree)

    def __repr__(self):
        return self.__str__()

    def contains_points(self, points):
        if not all([isinstance(p, Point) for p in points]):
            return False
        if not all([len(p.coordinates) == len(self.witness_points[0].coordinates) for p in points]):
            return False

        num_vars = len(self.witness_points[0].coordinates)
        codim = num_vars - self.dimension

        tmpdir = tempfile.mkdtemp()

        input_file = os.path.join(self.dirname, "input")
        witness_data_file = os.path.join(self.dirname, "witness_data")
        shutil.copy(witness_data_file, tmpdir) # copy over the witness_data file

        # copy over the input file, but change tracktype to 3
        lines = striplines(input_file)
        tracktype_specified = False
        for i in range(len(lines)):
            line = lines[i]
            if "tracktype" in line.lower():
                tracktype_specified = True
                lines[i] = "TrackType:3;"
        if not tracktype_specified:
            # in which I assume a config section; don't break this please
            lines.insert(1, "TrackType:3")

        input_file = os.path.join(tmpdir, "input")
        with open(input_file, "w") as fh:
            for line in lines:
                print(line, file=fh)

        # write out the start file (the points to test)
        member_points = os.path.join(tmpdir, "member_points")
        with open(member_points, "w") as fh:
            print(len(points), file=fh)
            for point in points:
                for coord in point.coordinates:
                    real = coord.real
                    imag = coord.imag
                    print("{0} {1}".format(real, imag), file=fh)
                print("", file=fh)

        run_bertini(tmpdir, "contains", screenout=False)

        # read the incidence_matrix file
        incidence_matrix = os.path.join(tmpdir, "incidence_matrix")
        lines = striplines(incidence_matrix)
        num_codims = int(lines[0])
        codim_lines = lines[1:num_codims+1]
        lines = lines[2+num_codims:] # num_points in the next line
        offset = 0
        for line in codim_lines:
            cod,numco = map(int, line.split(" "))
            if cod == codim:
                break
            offset += numco

        column = offset + self.component_id
        for line in lines:
            ismember = map(int, line.split(" "))
            if not bool(ismember[column]):
                return False

        return True

    def eq(self, other):
        cls = self.__class__
        if not isinstance(other, cls) or self.dimension != other.dimension or self.degree != other.degree: # obvious hole: num_variables
            return False

        self_in_other = other.contains_points(self.witness_points)
        other_in_self = self.contains_points(other.witness_points)

        return self_in_other and other_in_self
#       num_vars = len(self.witness_points[0].coordinates)
#       codim = num_vars - self.dimension
#       tmpdir = tempfile.mkdtemp()

#       # first test if other is contained in self
#       input_file = os.path.join(self.dirname, "input")
#       witness_data_file = os.path.join(self.dirname, "witness_data")
#       shutil.copy(witness_data_file, tmpdir) # copy over the witness_data file

#       # copy over the input file, but change tracktype to 3
#       lines = striplines(input_file)
#       tracktype_specified = False
#       for i in range(len(lines)):
#           line = lines[i]
#           if "tracktype" in line.lower():
#               tracktype_specified = True
#               lines[i] = "TrackType:3;"
#       if not tracktype_specified:
#           # in which I assume a config section; don't break this please
#           lines.insert(1, "TrackType:3")

#       input_file = os.path.join(tmpdir, "input")
#       with open(input_file, "w") as fh:
#           for line in lines:
#               print(line, file=fh)

#       # write out the start file (of the other guy's points)
#       other.write_witness_points(os.path.join(tmpdir, "member_points"))
#       run_bertini(tmpdir, "other_in_self", screenout=False)

#       # read the incidence_matrix file
#       incidence_matrix = os.path.join(tmpdir, "incidence_matrix")
#       lines = striplines(incidence_matrix)
#       num_codims = int(lines[0])
#       codim_lines = lines[1:num_codims+1]
#       lines = lines[2+num_codims:] # num_points in the next line
#       offset = 0
#       for line in codim_lines:
#           cod,numco = map(int, line.split(" "))
#           if cod == codim:
#               break
#           offset += numco

#       column = offset + self.component_id
#       for line in lines:
#           ismember = map(int, line.split(" "))
#           if not bool(ismember[column]):
#               return False

#       # now test if self is contained in other
#       input_file = os.path.join(other.dirname, "input")
#       witness_data_file = os.path.join(other.dirname, "witness_data")
#       shutil.copy(witness_data_file, tmpdir) # copy over the witness_data file

#       # copy over the input file, but change tracktype to 3
#       lines = striplines(input_file)
#       tracktype_specified = False
#       for i in range(len(lines)):
#           line = lines[i]
#           if "tracktype" in line.lower():
#               tracktype_specified = True
#               lines[i] = "TrackType:3;"
#       if not tracktype_specified:
#           # in which I assume a config section; don't break this please
#           lines.insert(1, "TrackType:3")

#       input_file = os.path.join(tmpdir, "input")
#       with open(input_file, "w") as fh:
#           for line in lines:
#               print(line, file=fh)

#       # write out the start file (of our points this time)
#       self.write_witness_points(os.path.join(tmpdir, "member_points"))
#       run_bertini(tmpdir, "self_in_other", screenout=False)

#       # read the incidence_matrix file
#       incidence_matrix = os.path.join(tmpdir, "incidence_matrix")
#       lines = striplines(incidence_matrix)
#       num_codims = int(lines[0])
#       codim_lines = lines[1:num_codims+1]
#       lines = lines[2+num_codims:] # num_points in the next line
#       offset = 0
#       for line in codim_lines:
#           cod,numco = map(int, line.split(" "))
#           if cod == codim:
#               break
#           offset += numco

#       column = offset + other.component_id
#       for line in lines:
#           ismember = map(int, line.split(" "))
#           if not bool(ismember[column]):
#               return False

#       # made it!
#       return True

    def write_witness_points(self, filename):
        with open(filename, "w") as fh:
            print(len(self.witness_points), file=fh)
            for point in self.witness_points:
                for coord in point.coordinates:
                    real = coord.real
                    imag = coord.imag
                    print("{0} {1}".format(real, imag), file=fh)
                print("", file=fh)

    @property
    def component_id(self):
        return self._component_id
    @property
    def degree(self):
        return self._degree
    @property
    def dirname(self):
        return self._dirname
    @property
    def dimension(self):
        return self._dimension
    @property
    def witness_points(self):
        return self._witness_points

def write_bertini_input_file(dirname, variable_group, constants, subfunctions, functions, parameters=None, pathvariables=None, options=collections.OrderedDict({"TrackType":1}), filename="input"):
    if not os.path.exists(dirname):
        os.mkdir(dirname) # throws an error when it should
    filename = os.path.join(dirname, filename)
    with open(filename, "w") as fh:
        print("CONFIG", file=fh)
        for k in options.keys():
            print("{0}:{1};".format(k, options[k]), file=fh)
        print("END;", file=fh)
        print("INPUT", file=fh)
        if "UserHomotopy" in options.keys():
            if options["UserHomotopy"] == 1:
                print("variable {0};".format(",".join(map(str, variable_group))), file=fh)
            elif options["UserHomotopy"] == 2:
                print("variable_group {0};".format(",".join(map(str, variable_group))), file=fh)
            else:
                print("Error!")
                exit()
        else:
            print("variable_group {0};".format(",".join(map(str, variable_group))), file=fh)
        print("function {0};".format(",".join(map(str, functions.keys()))), file=fh) # declare functions
        if pathvariables: # declare path variables
            print("pathvariable {0};".format(",".join(map(str, pathvariables))), file=fh)
        if parameters: # declare parameters
            print("parameter {0};".format(",".join(map(str, parameters.keys()))), file=fh)
        if constants: # declare constants
            print("constant {0};".format(",".join(map(str, constants.keys()))), file=fh)
        if parameters: # define parameters
            for k in parameters.keys():
                print("{0} = {1};".format(k, parameters[k]), file=fh)
        if constants: # define constants
            # print constants
            for k in constants.keys():
                print("{0} = {1};".format(k, constants[k]), file=fh)
        # print subfunctions
        if subfunctions:
            for k in subfunctions.keys():
                print("{0} = {1};".format(k, subfunctions[k]), file=fh)
        # print functions
        for k in functions.keys():
            print("{0} = {1};".format(k, functions[k]), file=fh)
        print("END;", file=fh)

def write_bertini_start_file(dirname, points, filename="start"):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filename = os.path.join(dirname, filename)
    with open(filename, "w") as fh:
        print(len(points), file=fh)
        for point in points:
            for coord in point.coordinates:
                real = coord.real
                imag = coord.imag
                print("{0} {1}".format(real, imag), file=fh)
            print("", file=fh)

def run_bertini(dirname, name, args=[], screenout=True):
    cwd = os.getcwd()
    os.chdir(dirname)

    if args:
        cmd = "bertini {0}".format(" ".join(args))
    else:
        cmd = "bertini"

    logfile = os.path.join(cwd, name + ".log")

    if screenout:
        cmd += " | tee " + logfile
    else:
        cmd += " >" + logfile
    os.system(cmd)

    os.chdir(cwd)

def striplines(filename):
    with open(filename, "r") as fh:
        lines = [l.strip() for l in fh.readlines() if l != "\n"]

    return lines

def read_main_data(dirname):
    filename = os.path.join(dirname, "main_data")
    lines = striplines(filename)

    num_variables = int(lines[0].split(": ")[1])
    rank = int(lines[2].split(": ")[1])
    lines = lines[3:]
    dims = {}
    coordinates = []
    for line in lines:
        if len(coordinates) == num_variables:
            dims[dim][component_number].append(Point(coordinates))
            coordinates = []

        if line.startswith("----------DIMENSION"):
            dim = int(line.split(" ")[1].strip("----------"))
            dims[dim] = {}
        elif line.startswith("Component number"):
            component_number = int(line.split(": ")[1])
            if component_number not in dims[dim].keys():
                dims[dim][component_number] = []
        elif line == "*************** input file needed to reproduce this run ***************":
            break
        else:
            try:
                real, imag = map(float, line.split(" "))
                coordinates.append(complex(real,imag))
            except ValueError:
                pass
        
    all_components = []
    dirname = os.path.abspath(dirname)
    for dim in dims.keys():
        dim_components = dims[dim]
        for component_id in dim_components.keys():
            witness_points = dim_components[component_id]
            component = IrreducibleComponent(witness_points, dim, component_id, dirname)
            all_components.append(component)

    return all_components

def read_solutions_file(dirname, filename="finite_solutions"):
    filename = os.path.join(dirname, filename)
    lines = striplines(filename)
    num_points = int(lines[0])
    lines = lines[1:]
    points = []

    num_coordinates = int(len(lines)/num_points)
    divided = [lines[i*num_coordinates:(i+1)*num_coordinates] for i in range(num_points)]
    for point in divided:
        point = Point([complex(*map(float, coord.split())) for coord in point])
        points.append(point)

    return points

