# In this case I'll do algorithm to minimize the energies cost
# in a data center usin Deep Q-Learning

# imports
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

mean_months = [6., 7., 9., 10., 14., 19., 23., 27., 21., 15., 11., 7.]

class Environment(object):
    
    def __init__(self, optim_temp = (18.0,24.0), init_mon = 0,
        init_num_users = 10, init_rate_data = 60):
        # initial parameters
        self.monthly_atmospheric_temp = mean_months
        self.init_mon = init_mon
        self.atmospheric_temp = self.monthly_atmospheric_temp[init_mon]
        self.optim_temp = optim_temp
        self.min_temp = -20
        self.min_users = 10
        self.min_rate_data = 5
        self.max_temp = 80
        self.max_users = 100
        self.max_update_users = 20
        self.max_update_data = 10
        self.max_rate_data = 300
        self.init_num_users = init_num_users
        self.current_num_users = init_num_users
        self.init_rate_data = init_rate_data
        self.current_rate_data = init_rate_data
        self.intrisic_temp = self.atmospheric_temp + 1.25\
            * self.current_num_users + 1.25 * self.current_rate_data
        self.ai_temp = self.intrisic_temp
        self.noai_temp = (self.optim_temp[0] + self.optim_temp[1] / 2.)
        self.ai_total_energy = 0.
        self.noai_total_energy = 0.
        self.reward = 0.
        self.game_over = 0
        self.train = 1

    def update_env(self, direction, energy_ai, month):
        # energy's calculate when don't use AI
        energy_noai = 0
        if (self.noai_temp < self.optim_temp[0]):
            energy_noai = self.optim_temp[0] - self.noai_temp
            self.noai_temp = self.optim_temp[0]
        elif (self.noai_temp > self.optim_temp[1]):
            energy_noai = self.noai_temp - self.optim_temp[1]
            self.noai_temp = self.optim_temp[1]

        # reward calculate and scale
        self.reward = energy_noai - energy_ai
        self.reward = 1e-3 * self.reward

        # obtain next stage

        # update temperatue
        self.atmospheric_temp = self.monthly_atmospheric_temp[month] 
        # update users numbers
        self.current_num_users +=\
            np.random.randint(-self.max_update_users,self.max_update_users)
        if (self.current_num_users > self.max_users):
            self.current_num_users = self.max_users
        elif (self.current_num_users < self.min_users):
            self.current_num_users = self.min_users
        # update rate
        self.current_rate_data +=\
            np.random.randint(-self.max_update_data,self.max_update_data)
        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data

        # calculating delta intrisic temperatue
        past_intrisic_temp = self.intrisic_temp
        self.intrisic_temp = self.atmospheric_temp + 1.25\
            * self.current_num_users + 1.25 * self.current_rate_data
        delta_intrisic_temp = self.intrisic_temp - past_intrisic_temp

        # calculating delta temperatue regulated by AI
        if (direction == -1):
            delta_temp_ai = -energy_ai
        elif (direction == 1):
            delta_temp_ai = energy_ai

        # update new server's temperature using AI
        self.ai_temp += delta_intrisic_temp + delta_temp_ai

        # update new server's temperature don't using AI
        self.noai_temp += delta_intrisic_temp

        # final check (game over)
        if (self.ai_temp < self.min_temp):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.ai_temp = self.optim_temp[0]
                self.ai_total_energy += self.optim_temp[0] - self.ai_temp
        elif (self.ai_temp > self.max_temp):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.ai_temp = self.optim_temp[1]
                self.ai_total_energy += self.ai_temp - self.optim_temp[1]
        
        # update target
        self.ai_total_energy += energy_ai
        self.noai_total_energy += energy_noai

        # scaling next state values
        scaled_temp_ai = (self.ai_temp - self.min_temp)\
            / (self.max_temp - self.min_temp)

        scaled_num_users = (self.current_num_users - self.min_users)\
            / (self.max_users - self.min_users)

        scaled_rate_data = (self.current_rate_data - self.min_rate_data)\
            / (self.max_rate_data - self.min_rate_data)

        next_state = np.matrix([scaled_temp_ai, scaled_num_users, scaled_rate_data])

        return next_state, self.reward, self.game_over

    # reset environment
    def reset(self, new_month):
        self.atmospheric_temp = self.monthly_atmospheric_temp[new_month]
        self.init_mon = new_month
        self.current_num_users = self.init_num_users
        self.current_rate_data = self.init_rate_data
        self.intrisic_temp = self.atmospheric_temp + 1.25\
            * self.current_num_users + 1.25 * self.current_rate_data
        self.ai_temp = self.intrisic_temp
        self.noai_temp = (self.optim_temp[0] + self.optim_temp[1]) / 2.
        self.ai_total_energy = 0.
        self.noai_total_energy = 0.
        self.reward = 0.
        self.game_over = 0.
        self.train = 1

    def obeserve(self):
        # scaling next state values
        scaled_temp_ai = (self.ai_temp - self.min_temp)\
            / (self.max_temp - self.min_temp)

        scaled_num_users = (self.current_num_users - self.min_users)\
            / (self.max_users - self.min_users)

        scaled_rate_data = (self.current_rate_data - self.min_rate_data)\
            / (self.max_rate_data - self.min_rate_data)

        next_state = np.matrix([scaled_temp_ai, scaled_num_users, scaled_rate_data])

        return next_state, self.reward, self.game_over

# brain
class Brain(object):

    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate

        # input layer
        states = Input(shape = (3,))

        # hidden layer
        x = Dense(units = 64, activation = 'relu')(states)
        y = Dense(units = 32, activation = 'relu')(x)

        # output layer
        q_values = Dense(units = number_actions, activation = 'softmax')(y)

        # creating model
        self.model = Model(inputs = states, outputs = q_values)

        # compile model
        self.model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))


# main of algorithm
if __name__ == '__main__':
    print("Main Algorithm")
    
