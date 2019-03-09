import numpy as np
import random
import time
import pygame
import sys
import math
import copy
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
#from keras.models import model_from_json
import line_profiler
from keras.optimizers import Adam, SGD
from pprint import pprint


#Valg som må tas:
#1. Hvordan representere players?
#2. Skal jeg ha at hver celle er et object eller bare la arrayet være og finne en annen måte å finne om noe er terminal?
#3. Teste om akkurat et move gjør at spillet har kommet til en terminal node vil nok være det som er mest effektivt. På denne måten vil man kun sjekke naboer til akkurat den brikken man nettopp la

BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (198, 226, 255)
COLORS = [WHITE, GREEN, LIGHT_BLUE, BLUE, RED]




class Game:
    def __init__(self, size, player, statemanager, anet, simulations, epsilon, screen_size=[640,640], visualize=False, caption="My Game"):
        
        self.manager = statemanager
        self.anet = anet
        self.board_size = size
        self.visualize = visualize
        self.n_x_n, self.current_node = self.manager.get_initial_state(size, player)
        
        
        if (self.visualize):
            pygame.init()
            pygame.display.set_caption(caption)
            self.screen = pygame.display.set_mode(screen_size)
            self.clock = pygame.time.Clock()        
            self.size = screen_size
            self.draw_game(self.n_x_n)
            ev = pygame.event.get()
            
        self.simulations = simulations
        self.epsilon = epsilon 
        

    def play(self):
        
        going = True
        while(True):
            while(going):                                 
                mcts = Mcts(self.current_node, self.manager, self.board_size, self.simulations, self.epsilon, self.anet)
                action, self.current_node = mcts.run()
                
                if(self.visualize):
                    action = self.manager.to_multi(action)
                    self.n_x_n[action[0]][action[1]] = self.current_node.player*-1
                    self.draw_game(self.n_x_n)
                    ev = pygame.event.get()
                    #time.sleep(1)

                if(self.manager.is_terminal(self.current_node)):
                    #print("Player " + str(self.current_node.player*-1) + " Won")
                    #return (self.current_node.player*-1)
                    going = False
                
                
            if(self.visualize):
                time.sleep(3)
                return (self.current_node.player*-1)
                
            else:
                return (self.current_node.player*-1)



    def rand_play(self, net_1, name_game, player):
        going = True
        while(True):
                         
            while(going):                                 
                if(self.current_node.player == player):
                    case = self.manager.to_network_board(self.current_node.grid, self.current_node.player)
                    action = net_1.predict(case, self.current_node.grid)
                else:
                    action = self.manager.get_random_move(self.current_node)
                
                self.current_node = self.manager.get_next_state(self.current_node, action)
                    
                    
                action = self.manager.to_multi(action)
                self.n_x_n[action[0]][action[1]] = self.current_node.player*-1

                self.draw_game(self.n_x_n)

                if(self.manager.is_terminal(self.current_node)):
                    going = False
                ev = pygame.event.get()
                time.sleep(0.7)
                
            if(self.visualize):
                print("The winner is: Player" + str(self.current_node.player*-1))
                pygame.image.save(self.screen, name_game)
                time.sleep(5)
                return (self.current_node.player*-1)
    
    
    
    
    def network_play(self, net_1, net_2, name_game, player):
        self.player = player
        going = True
        while(True):
                         
            while(going):                                 
                case = self.manager.to_network_board(self.current_node.grid, self.current_node.player)
                if(self.current_node.player == player):
                    action = net_1.predict(case, self.current_node.grid)
                else:
                    action = net_2.predict(case, self.current_node.grid)
                    
                self.current_node = self.manager.get_next_state(self.current_node, action)
                

                action = self.manager.to_multi(action)
                self.n_x_n[action[0]][action[1]] = self.current_node.player*-1

                self.draw_game(self.n_x_n)

                if(self.manager.is_terminal(self.current_node)):
                    going = False
                ev = pygame.event.get()
#!!!!!!!time.sleep(0.7)
                
            if(self.visualize):
                if (self.current_node.player*-1 == self.player):
                    print("The winner is: Player 1")
                else:    
                    print("The winner is: Player 2")
                pygame.image.save(self.screen, name_game)
                #time.sleep(1)
                return (self.current_node.player*-1)
        
    
      
    #From here down, only code for visualizing the board    
    def draw_game(self, grid):
        dot_positions = []
        start_x = int(self.size[0]/2)
        start_y = 50
        for rows in range(len(grid)):
            dot_positions.append([])
            y_pos = start_y + rows*50
            x_pos = start_x - rows*50
            for column in range(len(grid[rows])):
                y_pos = y_pos + math.ceil(column/len(grid))*50
                x_pos = x_pos + math.ceil(column/len(grid))*50
                dot_positions[rows].append((x_pos, y_pos))
        
        self.draw_lines(dot_positions)
        pygame.draw.lines(self.screen, GREEN, False, [((self.size[0]/2 + 50),50), ((self.size[0]/2+(self.board_size-2)*50+50), 50*(self.board_size-1))], 5)
        pygame.draw.lines(self.screen, RED, False, [((self.size[0]/2-(self.board_size-2)*50-50), 50*(self.board_size-1)),((self.size[0]/2 - 50),50)], 5)
        
        pygame.draw.lines(self.screen, GREEN, False, [((self.size[0]/2-(self.board_size-2)*50-50), (self.board_size*50+50)),((self.size[0]/2-50),(((self.board_size)*2)*50-50))], 5)
        pygame.draw.lines(self.screen, RED, False, [((self.size[0]/2 + 50), ((self.board_size)*2)*50-50),((self.size[0]/2+(self.board_size-2)*50+50),50*(self.board_size-1)+100)], 5)
        #pygame.draw.lines(self.screen, GREEN, False, [100, 150], 1)
        
        for i in range(len(dot_positions)):
            for pos in range(len(dot_positions[i])):
                color = COLORS[self.n_x_n[i][pos]]
                self.draw_circle(dot_positions[i][pos], color)
                
        pygame.display.update()
        
    def draw_circle(self, pos, color):
        pygame.draw.circle(self.screen, color, pos, 10)
    
    def draw_line(self, arr):
        pygame.draw.lines(self.screen, WHITE, False, arr, 1)
    

    def draw_lines(self, array):
        for row in range(len(array)):
            for column in range(len(array[row]) - 1):
                stop_hor = array[row][column]
                start_hor = array[row][column + 1]
                
                stop_ver = array[column][row]
                start_ver = array[column + 1][row]
                
                self.draw_line([start_hor,stop_hor])
                self.draw_line([start_ver, stop_ver])
                self.draw_line([stop_hor, stop_ver])


class Topp:
    def __init__(self, k, saving, mbs, optimizer, stateman, g, player, size, screen_size):
        self.k = k
        self.saving = saving
        self.optimizer = optimizer
        self.statemanager = stateman
        self.g = g
        self.player = player
        self.mbs = mbs
        self.size = size
        self.init_players()
        self.screen_size = screen_size
        
        
        
        
    def init_players(self):
        self.players = {}
        self.score = {}
        for i in range(self.k+1):
            self.players["player " + str(i)] = Anet(self.statemanager)
            self.score["player " + str(i)] = 0
            self.players["player " + str(i)].load_model("models/vise_frem/model_" + str(int(self.saving*i))+".h5")
        #print(self.players)
        
        
                
        
    def play(self):
        player = self.player
        for ga in range(self.g):
            for i in range(self.k):
                for j in range(i+1, (self.k+1)):
                    print("\n---------------Changing players---------------")
                    print("Player 1: Model_" + str(int(i*self.saving)) + " VS Player 2: Model_" + str(int(j*self.saving)))
                    #time.sleep(1)
                    if(player == 1):
                        caption = ("Green: M_" + str(int(i*self.saving)) + " VS Red: M_" + str(int(j*self.saving)))
                    else:
                        caption = ("Red: M_" + str(int(i*self.saving)) + " VS Green: M_" + str(int(j*self.saving)))
                    
                    game = Game(self.size, player, self.statemanager, anet=None, simulations=None, epsilon=None, visualize=True, caption=caption, screen_size = self.screen_size) 
                    winner = game.network_play(self.players["player " + str(i)], self.players["player " + str(j)], ("game_pictures/"+str(caption) + ", "+ str(ga)), player)
                    if(ga%2 == 0 and winner == 1):
                        self.score["player " + str(i)] += 1
                    if(ga%2 == 0 and winner == -1):
                        self.score["player " + str(j)] += 1
                    if(ga%2 == 1 and winner == -1):
                        self.score["player " + str(i)] += 1
                    if(ga%2 == 1 and winner == 1):
                        self.score["player " + str(j)] += 1
            player = player * (-1)
            
        print("\n-------------------SCORE-------------------")
        print(self.score)
                
                
        

class Stateman:

    def __init__(self, size):
        self.size = size
        self.neighbors = {}
        self.get_neighbors()
        self.p1_start_indices = []
        self.p1_end_indices = []
        self.p2_start_indices = []
        self.p2_end_indices = []
        self.player = {-1: (self.p2_start_indices, self.p2_end_indices), 1: (self.p1_start_indices, self.p1_end_indices)}
        self.indices()
        self.double = self.size*self.size
        self.replay_buffer = deque(maxlen=1000)
        
    def get_move(self, prediction):
        action = [int(prediction/self.size),int(prediction%self.size)]
        return action
    
    def add_to_replay_buffer(self, case):
        self.replay_buffer.appendleft(case)
    
        
    def get_cases(self, mbs):
        #Må fikse slik at den tar random og ikke går direkte i rekkefølge. Velger også nå alle istedenfor en batch?
        end = len(self.replay_buffer)
        arr = list(self.replay_buffer)
        x = []
        y = []
        randnums = random.sample(range(0, end), min(mbs, end))
        for num in randnums:
            x.append(arr[num][0])
            y.append(arr[num][1])
        return x,y
        
        
    def indices(self):
        twice = self.size*self.size
        for i in range(self.size):
            self.p1_start_indices.append(i)
            self.p1_end_indices.append(twice - (i+1))
            self.p2_start_indices.append(self.size*i)
            self.p2_end_indices.append((self.size-1) + i*self.size)
    
    def intersection(self, a, b):
        A = set(a)
        B = set(b)
        return (A & B)
        
       
    def get_neighbors(self): 
        neighbors = []
        allowed_neighbors = []
        for r in range(self.size):
            for c in range(self.size):
                allowed_neighbors.append(self.to_one([r,c]))
                neighbors.append([self.to_one([r-1,c]), self.to_one([r-1, c+1]), self.to_one([r, c+1]), self.to_one([r+1, c]), self.to_one([r+1, c-1]), self.to_one([r, c-1])])


        for i in range(self.size*self.size):
            self.neighbors[i] = (self.intersection(allowed_neighbors, neighbors[i]))
        
        
    def get_initial_state(self, size, player):
        grid = [0]*(size*size)
        grid2 = []
        network_board = [0]*(self.double)*3
        if(player == 1):
            network_board[:self.double] = [1]*self.double
        else:
            network_board[:self.double] = [0]*self.double
            
        for i in range(size):
            grid2.append([0]*size)
        return grid2, Node(grid, player, network_board, seen=True)

        
    def get_available_actions(self, node):
        return np.where(np.array(node.grid) == 0)        
         

    def get_next_state(self, node, action):
        grid = node.grid.copy()
        grid[action] = node.player
        
        network = node.network_board.copy()
       # print(network)
        if(node.player == 1):
            network[:self.double] = [1]*self.double
            network[(action*2)+self.double] = 0
            network[(action*2)+(self.double+1)] = 1 
        else:
            network[:self.double] = [0]*self.double
            network[(action*2)+self.double] = 1
            network[(action*2)+(self.double+1)] = 0
            
            
        return Node(grid, node.player*-1, network, action, parent=node)
    
    def get_random_move(self, node):
        available_actions = self.get_available_actions(node)[0]
        random_move = random.randint(0,len(available_actions)-1)
        action = available_actions[random_move]
        return action
    
    def add_child(self, node, child):
        node.children.append(child)
    
    def add_visits_and_win(self, value, node):
        node.num_visits += 1
        node.total_wins += value
    
    
    def rolled_out(self, node):
        node.rollout = True
    
    #@profile
    def is_terminal(self, node):
        start_indices, end_indices = self.player[node.player*(-1)]
        explore = []
        visited = []
        player = node.player*-1
        
        for i in start_indices:
            if(node.grid[i] == player):
                explore.append(i)
        
        while len(explore) != 0:
            current_node = explore.pop()
            if(current_node in end_indices):
                return True
            
            if(current_node in visited):
                continue
            for neigh in self.neighbors[current_node]:
                if(node.grid[neigh] == player):
                    explore.append(neigh)
                    
            visited.append(current_node)
        
        return False
    
    
    def to_network_board2(self, grid, player):
        double = self.size*self.size
        grid2 = [0]*((double)*3)
        if(player == 2):
            grid2[:double] = [0]*double
        else:
            grid2[:double] = [1]*double
        
        for i in range(len(grid)):
            if(grid[i] == 2):
                grid2[(i*2)+double] = 1
                grid2[(i*2)+(double+1)] = 0
            elif(grid[i] == 1):
                grid2[(i*2)+double] = 0
                grid2[(i*2)+(double+1)] = 1
        return grid2
        
    
    
    
    def to_network_board(self, grid, player):
        double = self.size*self.size
        grid2 = [0]*((double)*3)
        if(player == -1):
            grid2[:double] = [0]*double
        else:
            grid2[:double] = [1]*double
        
        for i in range(len(grid)):
            if(grid[i] == -1):
                grid2[(i*2)+double] = 1
                grid2[(i*2)+(double+1)] = 0
            elif(grid[i] == 1):
                grid2[(i*2)+double] = 0
                grid2[(i*2)+(double+1)] = 1
        return grid2
    
    def make_arr(self):
        return [0]*(self.size*self.size)
    
        
    #@profile
    def to_one(self, arr):
        if(arr[0] < 0 or arr[1] < 0 or arr[0] > (self.size-1) or arr[1] > (self.size-1)):
            return -100
        return (self.size*arr[0] + arr[1])
    
    def to_multi(self, num):
        return [int(num/self.size), num%self.size]
        
        #Use DFS to search the neighbors of the piece that is put with the move. Have to account that not all places have all 6 neighbors. Also have to know if a certain place is a border place and which player this border belongs to. 
        
class Mcts():
    
    def __init__(self, root, stateman, size, simulations, epsilon, anet):
        self.current_state = root
        self.stateman = stateman
        self.d = self.stateman.make_arr()
        self.size = size
        self.simulations = simulations
        self.epsilon = epsilon
        self.anet = anet
     
    def run(self):
        for i in range(self.simulations):
            self.tree_policy(self.current_state)
       
        chosen_node = []
        
        for child in self.current_state.children:
            chosen_node.append(child.num_visits)
            place = child.action
            self.d[place] = child.num_visits/self.current_state.num_visits
        
        action = np.argmax(np.array(chosen_node))

        #Might be better to use two list for the replay_buffer. This way I can use np.arrays from the beginning instead of going back and forth.
        grid = self.current_state.grid.copy()
        #grid.insert(0,self.current_state.player)
        
    
        grid2 = self.stateman.to_network_board(grid, self.current_state.player)
        
        
        self.stateman.add_to_replay_buffer([grid2, self.d])
        return self.current_state.children[action].action, self.current_state.children[action]
    
    
    def tree_policy(self, start_state):
        current_state = start_state
        while(current_state.children):
            next_state = self.selection(current_state)
            current_state = next_state
        
        if(current_state.rollout):
            self.node_expansion(current_state)
        else:
            self.rollout(current_state)
        
        
    
    
    def selection(self, state):
        poss_next_state = []

        for child in state.children:
            #print(child)
            if(child.num_visits == 0):
                if(state.player == 1):
                    poss_next_state.append(9999)
                    #print("Kjørte denne og?")
                elif(state.player == -1):
                    poss_next_state.append(-9999)
            
            else:
                if(state.player == 1):
                    poss_next_state.append(((child.total_wins/child.num_visits) + (4 * math.sqrt((math.log(state.num_visits))/(1+child.num_visits)))))
                elif(state.player == -1):
                    poss_next_state.append(((child.total_wins/child.num_visits) - (4 * math.sqrt((math.log(state.num_visits))/(1+child.num_visits)))))
                             
        if(state.player == 1):
            next_state = state.children[np.argmax(np.array(poss_next_state))]
        elif(state.player == -1):
            next_state = state.children[np.argmin(np.array(poss_next_state))]
        return next_state
    
    
    
    def node_expansion(self, state):
        #Check to see if the current state is a terminal state.
        if(self.stateman.is_terminal(state)):
            if(state.player == 1):
                #If the player is player 1 in this state it means player 2 won. Being the player at a terminal state means
                self.backprop(-2, state)
            elif state.player == -1:
                self.backprop(2, state)
            return
            
        #Get possible actions from this state.
        actions = self.stateman.get_available_actions(state)
        for action in actions[0]:
            child = self.stateman.get_next_state(state, action)
            self.stateman.add_child(state, child)

        #Choose the first of the new children as the current state
        next_state = state.children[0]
        
        self.rollout(next_state)
            
        
    
    def rollout(self, current_state):
        root = current_state
        current_node = current_state
        while(not self.stateman.is_terminal(current_node)):
            #Sannsynlighet for at man velger random move er epislon. Starter som 1 og minker ettersom nettverket blir smartere. 
            if(random.random() < self.epsilon):
                action = self.stateman.get_random_move(current_node)
            else:
                action = self.anet.predict(current_node.network_board, current_node.grid)
            
            current_node = self.stateman.get_next_state(current_node, action)
            
        winner = current_node.player*(-1)
        self.stateman.rolled_out(root)
        #Setting the value to be added to the states when hitting a terminal state
        self.backprop(winner, root)
    
    
    #Fix so I dont update wins on root nodea
    def backprop(self, winner, current_state):
        current_node = current_state
        self.stateman.add_visits_and_win(winner, current_node)
        #adding a value to the state. Either it is a win for player 1 => +1 or a win for player 2 => -1
        
        while (current_node.parent):
            current_node = current_node.parent
            self.stateman.add_visits_and_win(winner, current_node)
        return    
    
            



class Node:
    def __init__(self, grid, player, network, action=None, parent=None, seen=False):
        self.grid = grid
        self.player = player
        self.action = action
        self.network_board = network
        
        self.parent = parent
        self.children = []
        self.num_visits = 0
        self.total_wins = 0
        self.rollout = seen
        
        
        
        
        
class Anet():
    def __init__(self, manager):
        self.manager = manager
    
    
    
    def init_model(self, output_size, input_size, optimizer, layers, neurons, activation):
        self.model = Sequential()
        #The input layer to the neural network. It has here 2 input dimensions which leads to 50 neurons with relu activation
        self.model.add(Dense(neurons[0], input_dim=input_size, activation=activation[0]))
        
        for i in range(1,layers):
            self.model.add(Dense(neurons[i], activation=activation[i]))
        #self.model.add(Dense(75, activation='relu'))
            
        #This is the output layer with only 1 output and a sigmoid activation function to normalize it. 
        self.model.add(Dense(output_size, activation=activation[-1]))
        #
        self.model.compile(loss='mean_squared_error', optimizer=Adam())
        
        
    def load_model(self, name):
        self.model = load_model(name)    
        
    def save_model(self, num):
            self.model.save('models/demo/model_' + str(num)+ ".h5")        
    
    def fit(self, mbs):
        x,y = self.manager.get_cases(mbs)     
        x = np.array(x) 
        y = np.array(y)  
        self.model.train_on_batch(x, y)
        #self.model.fit(x=x, y=y, batch_size=mbs)
        #scores = self.model.evaluate(np.array(x), np.array(y), verbose=0)

        
    def predict(self, grid, mapping):
        case = np.array([grid])
        prediction = self.model.predict(case)
        prediction = self.normalize(mapping, prediction[0])
        action = np.argmax(prediction)
        return action
        
        
    def normalize(self, case, prediction):
        non_zero = np.nonzero(case)[0]

        for ele in non_zero:
            prediction[ele] = 0
        total = np.sum(prediction)
        factor = 1/total
        prediction = prediction * factor
        return prediction    
    
    


def main():
    statistics = 0                  #Variable to keep track of number of wins by player 1
    

    size = 5                        #Size of the n*n board. Gives a board of size*size
    screen_size = [640, 640]        #Screen size of the visualized board
   # visualization = False          #If the game should be visualized or not
    statemanager = Stateman(size)   #Initialization of the statemanager
    epsilon = 1                     #Starting value of epsilon. The probability of choosing rollout move based on the neural network.
    
    #Game related:
    player = 1                      #The starting player
    num_games = 1000                 #How many games it should play
    simulations = 10               #Number of monte carlo simulations
    k = 6                           #Number times the neural network should be saved
    mbs = 32                        #Number of cases taken from the replay buffer
    

    g = 50                           #Number of games played by the networks in TOPP
    
    
    input_size  = (size*size)*3     #The input size to the neural network should equal board state + one value for the player 
    output_size = size*size         #The size of the output layer from the neural network should equal all possible moves on the board
    number_of_layers = 4            #The number of hidden layers
    neurons_hidden_layer = [100,200,25] #Brukes ikke ATM!     #The neurons per hidden layer
    learning_rate = 0.001           #Brukes ikke ATM!      #The learning rate of the neural network
    activation = ['relu', 'relu', 'tanh', 'softmax']
    optimizer = Adam(lr=learning_rate)
    anet = Anet(statemanager)  #Initialization of the neural network
    
    
    
    
    saving = num_games/(k-1)            #The interval for saving the state of the network
    training = False
    topp = True
    #"-----------TRAINING-----------"
    if(not topp):
        if(training):
            anet.init_model(output_size, input_size, optimizer, number_of_layers, neurons_hidden_layer, activation)
            anet.save_model(0)
            for i in range(1, num_games+1):
                game = Game(size, player, statemanager, anet, simulations, max(epsilon,0.7), visualize=True, screen_size=screen_size)
                winner = game.play()
                if(winner == 1):
                    statistics += 1
                epsilon = epsilon * 0.99
                player = player * -1
                anet.fit(mbs)
                if((i)%saving == 0):
                    anet.save_model(i)
                    print("Model saved at " + str(i))
                print("Player 1 have won: " + str(statistics) + " of " + str(i) +  ", " + str(statistics/(i)*100) + "%")
        
        else:
            anet.load_model("models/vise_frem/model_1000.h5")
            game = Game(size, player, statemanager, anet, simulations, epsilon, visualize=True)
            game.rand_play(anet, "Random_play.jpg", player)
        
    else:   
        topp = Topp(k-1, saving, mbs, optimizer, statemanager, g, player, size, screen_size)
        topp.play()
    


#Different optimizers:
#1.SGD
#2.RMSprop
#3.Adagrad
#4.Adam


if __name__ == "__main__":
    main()
