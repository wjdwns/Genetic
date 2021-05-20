from math import sqrt
import math, random
import csv

# Models a city
class City:
    def __init__(self, x=None, y=None, distance=None,num =None):
        self.x = x
        self.y = y
        self.distance = None
        self.num = num

        if distance is not None:
            self.distance = distance
        else:
            self.distance = sqrt(self.x**2+self.y**2)

    # Gets city's x(y) coordinate
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    # Gets the distance to given city
    def distanceTo(self, city):
        xDistance = abs(self.getX() - city.getX())
        yDistance = abs(self.getY() - city.getY())
        distance = math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
        return distance

    def __repr__(self):
        return str(self.num)

    def getAbsoluteDist(self, i):
        if i < 25:
            dist = sqrt((self.x - (i * 4)) ** 2 + (self.y) ** 2)
        else:
            dist = sqrt((self.x) ** 2 + (self.y - (i * 4)) ** 2)
        return dist

# Holds the cities of a tour
class TourManager:
    # Holds our cities
    destinationCities = []
    # Adds a destination city
    def addCity(self, city):
        self.destinationCities.append(city)

    # Get a city
    def getCity(self, index):
        return self.destinationCities[index]

    # Get the number of destination cities
    def numberOfCities(self):
        return len(self.destinationCities)



# Stores a candidate tour
class Tour:
    # Constructs a blank tour
    def __init__(self, tourmanager, tour=None):
        # Holds our tour of cities
        self.tourmanager = tourmanager
        self.tour = []

        # Cache
        self.fitness = 0.0
        self.distance = 0

        # Constructs a blank tour
        if tour is not None:
            self.tour = tour
        else:
            for i in range(0, self.tourmanager.numberOfCities()):
                self.tour.append(None)

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, index):
        return self.tour[index]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        data = ['0']
        geneString = 'Start -> '
        for i in range(0, self.tourSize()):
            geneString += str(self.getCity(i)) + ' -> '
        geneString += 'End'
        with open('solution.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            for i in range(0, self.tourSize()):
                data[0] = str(self.getCity(i))
                writer.writerow((data))
        return geneString

    # Creates a random individual
    def generateIndividual(self, i):
        for cityIndex in range(0, self.tourmanager.numberOfCities()):
            self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
        self.tour = sorted(self.tour,key=lambda x:x.distanceTo(self.tourmanager.getCity(cityIndex)))

    # Gets a city from the tour
    def getCity(self, tourPosition):
        return self.tour[tourPosition]

    # Sets a city in a certain position within a tour
    def setCity(self, tourPosition, city):
        self.tour[tourPosition] = city
        self.fitness = 0.0
        self.distance = 0

    # Gets the tours fitness
    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.getDistance())
        return self.fitness

    # Gets the total distance of the tour
    def getDistance(self):
        if self.distance == 0:
            tourDistance = 0
            for cityIndex in range(0, self.tourSize()):
                fromCity = self.getCity(cityIndex)
                destinationCity = None
                if cityIndex + 1 < self.tourSize():
                    destinationCity = self.getCity(cityIndex + 1)
                else:
                    destinationCity = self.getCity(0)
                tourDistance += fromCity.distanceTo(destinationCity)
            self.distance = tourDistance
        return self.distance

    # Get number of cities on our tour
    def tourSize(self):
        return len(self.tour)

    # Check if the tour contains a city
    def containsCity(self, city):
        return city in self.tour


# Manages a population of candidate tours
class Population:
    # Construct a population
    def __init__(self, tourmanager, populationSize, initialise):
        # Holds population of tours
        self.tours = []
        for i in range(0, populationSize):
            self.tours.append(None)

        # If we need to initialise a population of tours do so
        if initialise:
            # Loop and create individuals
            for i in range(0, populationSize):
                newTour = Tour(tourmanager)
                newTour.generateIndividual(i)
                self.saveTour(i, newTour)

    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    # Saves a tour
    def saveTour(self, index, tour):
        self.tours[index] = tour

    # Gets a tour from population
    def getTour(self, index):
        return self.tours[index]

    # Gets the best tour in the population
    def getFittest(self):
        fittest = self.tours[0]

        # Loop through individuals to find fittest
        for i in range(0, self.populationSize()):
            if fittest.getFitness() <= self.getTour(i).getFitness():
                fittest = self.getTour(i)
        return fittest

    # Gets population size
    def populationSize(self):
        return len(self.tours)


# Manages algorithms for evolving population
class GA:
    def __init__(self, tourmanager, mutationRate=0.01, tournamentSize=25, elitism=True):
        self.tourmanager = tourmanager
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.elitism = elitism

    # Evolves a population over one generation
    def evolvePopulation(self, pop):
        newPopulation = Population(self.tourmanager, pop.populationSize(), False)

        # Keep our best individual if elitism is enabled
        elitismOffset = 0
        if self.elitism:
            newPopulation.saveTour(0, pop.getFittest())
            elitismOffset = 1

        # Crossover population
        # Loop over the new population's size and create individuals from
        # Current population
        numOfPos = 1
        for i in range(elitismOffset, newPopulation.populationSize()):
            # Select parents
            parent1 = self.tournamentSelection(pop)
            parent2 = self.tournamentSelection(pop)
            # Crossover parents
            #if i > newPopulation.populationSize()/1.5:
            #    numOfPos = 0
            child = self.crossover(parent1, parent2, numOfPos)
            # Add child to new population
            newPopulation.saveTour(i, child)

        # Mutate the new population a bit to add some new genetic material
        for i in range(elitismOffset, newPopulation.populationSize()):
            if i < newPopulation.populationSize()*0.70:
                self.mutate(newPopulation.getTour(i))
            else:
                self.mutate_dyn(newPopulation.getTour(i))

        return newPopulation

    # Applies crossover to a set of parents and creates offspring
    def crossover(self, parent1, parent2, numOfPos):
        # Create new child tour
        child = Tour(self.tourmanager)

        numOfPos = int(random.random() * 2)
        #numOfPos = 0
        #numOfPos = 1

        if numOfPos:#numOfPos = 1
            Pos = int(random.random() * parent1.tourSize())

            for i in range(0, Pos):
                child.setCity(i, parent1.getCity(i))

            for i in range(0, child.tourSize()):
                if not child.containsCity(parent2.getCity(i)):
                    # Loop to find a spare position in the child's tour
                    for ii in range(0, child.tourSize()):
                        # Spare position found, add city
                        if child.getCity(ii) == None:
                            child.setCity(ii, parent2.getCity(i))
                            break

        else:#numOfPos = 0
            # Get start and end sub tour positions for parent1's tour
            startPos = int(random.random() * parent1.tourSize())
            endPos = int(random.random() * parent1.tourSize())

            # Loop and add the sub tour from parent1 to our child
            for i in range(0, child.tourSize()):
                # If our start position is less than the end position
                if startPos < endPos and i > startPos and i < endPos:
                    child.setCity(i, parent1.getCity(i))
                # If our start position is larger
                elif startPos > endPos:
                    if not (i < startPos and i > endPos):
                        child.setCity(i, parent1.getCity(i))
            # Loop through parent2's city tour
            for i in range(0, parent2.tourSize()):
                # If child doesn't have the city add it
                if not child.containsCity(parent2.getCity(i)):
                    # Loop to find a spare position in the child's tour
                    for ii in range(0, child.tourSize()):
                        # Spare position found, add city
                        if child.getCity(ii) == None:
                            child.setCity(ii, parent2.getCity(i))
                            break

        return child

    #
    def mutate(self, tour):#초반 돌연변이
        # Loop through tour cities
        for tour1 in range(0, tour.tourSize()):
            # Apply mutation rate
            # 0.05 랜덤은 0부터 1사이의 값을 랜덤으로 생성함
            if random.random() < self.mutationRate:
                # Get a second random position in the tour
                tour2 = int(tour.tourSize() * random.random())

                # Get the cities at target position in tour
                city1 = tour.getCity(tour1)
                city2 = tour.getCity(tour2)

                # Swap them around
                tour.setCity(tour2, city1)
                tour.setCity(tour1, city2)

    def mutate_dyn(self, tour):#후반 돌연변이
        for tour1 in range(0, tour.tourSize()):
            # Apply mutation rate
            if random.random() < (self.mutationRate-0.005): #0.05 랜덤은 0부터 1사이의 값을 랜덤으로 생성함
                # Get a second random position in the tour
                tour2 = int(tour.tourSize() * random.random())

                # Get the cities at target position in tour
                city1 = tour.getCity(tour1)
                city2 = tour.getCity(tour2)

                # Swap them around
                tour.setCity(tour2, city1)
                tour.setCity(tour1, city2)
    # Selects candidate tour for crossover


    # Selects candidate tour for crossover
    def tournamentSelection(self, pop):
        # Create a tournament population
        tournament = Population(self.tourmanager, self.tournamentSize, False)

        #  For each place in the tournament get a random candidate tour and add it
        for i in range(0, self.tournamentSize):
            randomId = int(random.random() * pop.populationSize())
            tournament.saveTour(i, pop.getTour(randomId))
        #  Get the fittest tour
        fittest = tournament.getFittest()
        return fittest


# Create a tour and evolve a solution
if __name__ == '__main__':
    cities = []
    # 경로는 본인에 맞게 수정
    with open('TSP.csv', mode='r', newline='')as tsp:

        reader = csv.reader(tsp)
        for row in reader:
            cities.append(row)

        n_cities = 1000
        population_size = 50
        n_generations = 2000
        random.seed(100)

        # Setup cities and tour
        tourmanager = TourManager()

        for i in range(n_cities):
            x = float(cities[i][0])
            y = float(cities[i][1])
            tourmanager.addCity(City(x=x, y=y,num=i))
    tourmanager
    # Initialize population
    pop = Population(tourmanager, populationSize=population_size, initialise=True)
    print("Initial distance: " + str(pop.getFittest().getDistance()))

    # Evolve population for 50 generations
    ga = GA(tourmanager)
    pop = ga.evolvePopulation(pop)
    for i in range(n_generations):
        pop = ga.evolvePopulation(pop)
        fittest = pop.getFittest()

    # Print final results
    print("Finished")
    print("Final distance: " + str(pop.getFittest().getDistance()))
    print ("Solution:")
    print (pop.getFittest())