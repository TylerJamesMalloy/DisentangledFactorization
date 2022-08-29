import scipy as sc
import scipy.io
import scipy.stats

import scipy.optimize as optimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

import numpy as np
import math 
import random 

class frl_env: 
    actions = []
    states = []
    rewards = []

    def __init__(self, states=[], actions=[], rewards=[]):
        self.actions = actions 
        self.states = states 
        self.rewards = rewards 

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_rewards(self):
        return self.rewards
    

class Feature_RL:
    env = frl_env()
    eta = 0
    delta = 0
    beta = 0
    def __init__(self, env, eta = .9, delta = 0, beta = 15):
        # eta = learning rate 
        self.env = env
        self.q_table = np.zeros(self.env.get_states().shape)
        self.beta = beta
        self.beta_original = beta
        self.state = 0
        self.eta = eta
        self.eta_original = eta
        self.delta = delta
        self.delta_original = delta

    def get_q_table(self): 
        return self.q_table
    
    def get_env(self): 
        return self.env
    
    def get_policy(self): 
        return self.policy

    def get_state(self): 
        return self.state 
    
    def get_softmax(self):
        return self.beta

    def train_state_reward(self, state, reward):
        sum = 0
        for i in range(0, len(self.q_table[0])):
            sum += self.q_table[i][state[i] - 1]

        for i in range(0, len(self.q_table[0])):
            self.q_table[i][state[i] - 1] += self.eta * (reward - sum)
            #self.q_table[i][state[i] - 1] += self.eta * (reward - self.q_table[i][state[i] - 1])
        return 0

    def forget(self, chosen):
        for option_index in range(0,len(self.q_table)):
            for feature_index in range(0,len(self.q_table[0])):
                if(feature_index != (chosen[option_index] - 1)):
                    self.q_table[option_index][feature_index] *= (1- self.delta)
                
    def predict(self,states):
        value_one = 0
        value_two = 0
        value_three = 0
        
        try:
            value_one = math.exp(self.beta * self.state_value(states[0]))
        except OverflowError:
            value_one = 0

        try:
            value_two = math.exp(self.beta * self.state_value(states[1]))
        except OverflowError:
            value_two = 0
        
        try:
            value_three = math.exp(self.beta * self.state_value(states[2]))
        except OverflowError:
            value_three = 0

        state_values = [
            value_one, 
            value_two, 
            value_three
        ] 

        if(state_values == [0,0,0]):
            state_values = [1/3,1/3,1/3]

        # prediction_state = np.argmax(state_values) 
        # prediction = states[prediction_state]

        state_softmax = [
            state_values[0] / sum(state_values),
            state_values[1] / sum(state_values),
            state_values[2] / sum(state_values)
        ]

        guess = np.random.choice([0,1,2], 1, p=[state_softmax[0], state_softmax[1], state_softmax[2]])[0]
        prediction = states[guess]

        return (prediction, guess, [state_softmax[0], state_softmax[1], state_softmax[2]])
    
    def state_value(self, state):
        state_color_q = self.q_table[0][state[0] - 1]
        state_shape_q = self.q_table[1][state[1] - 1]
        state_texture_q = self.q_table[2][state[2] - 1]

        return state_color_q + state_shape_q + state_texture_q

    def reset(self):
        self.q_table = np.zeros(self.env.get_states().shape)
        self.eta = self.eta_original
        self.beta = self.beta_original
        self.delta = self.delta_original
        self.state = 0
        return 0
    
    # log prior no limit on search
    def get_thetas():
        thetas_500 =[
            [ 0.052570623045241405,8.75493124508224 ,0],
            [ 0.10740044231495904,5.312949771333232 ,0],
            [ 0.03736024636640811,19.471193487619406 ,0],
            [ 0.0707758138461334,9.861336476025475 ,0],
            [ 0.02724011209743759,15.527527698240672 ,0],
            [ 0.03371668443281594,14.541298000504295 ,0],
            [ 0.021797951448855605,25.27236105599262 ,0],
            [ 0.043864150589191166,10.90864386724759 ,0],
            [ 0.021361856989143713,20.17130093121118 ,0],
            [ 0.10020166488352103,4.587586405479023 ,0],
            [ 0.01974273160153416,14.352456410392985 ,0],
            [ 0.05589367296549495,14.636823139685003 ,0],
            [ 0.06574131568107697,7.728847688689287 ,0],
            [ 0.027028295513925344,12.486121870498428 ,0],
            [ 0.020494389462840962,31.20906551517967 ,0],
            [ 0.030969949183735402,18.41105349688079 ,0],
            [ 0.11680950711806745,7.343706279291198 ,0],
            [ 0.04148125170595236,16.293580493042708 ,0],
            [ 0.027500505615618606,16.258681787625846 ,0],
            [ 0.025656317990234875,19.243606408531647 ,0],
            [ 0.028042378228394324,15.394575959337875 ,0],
            [ 0.04933284999883158,16.410300689206245 ,0]
        ]

        thetas_300 = [
            [ 0.11118416352645771,6.400200853405744 ,0],
            [ 0.03877419353996027,5.315784180074056 ,0],
            [ 0.05220414843543956,13.195165016241448 ,0],
            [ 0.0987985607700613,7.6742544952148 ,0],
            [ 0.1292214990368296,7.080915851573841 ,0],
            [ 0.04660805728068766,10.672832221877105 ,0],
            [ 0.02445489327185942,21.960582390665678 ,0],
            [ 0.14679514723408224,6.331547358438375 ,0],
            [ 0.06248792691124732,12.789172198078948 ,0],
            [ 0.036503851369921715,9.859686622223757 ,0],
            [ 0.01910919196266764,15.050912535288655 ,0],
            [ 0.1202493997326275,11.135786685627698 ,0],
            [ 0.040782804866294105,12.414047345588669 ,0],
            [ 0.035900302943598694,11.0380298301121 ,0],
            [ 0.04024621240096056,16.453405476117155 ,0],
            [ 0.09641007549214815,7.196848639029903 ,0],
            [ 0.04669381079322325,16.362203331186638 ,0],
            [ 0.08606901685222607,10.011622732120438 ,0],
            [ 0.04663503828198235,15.252833389979813 ,0],
            [ 0.043331142069875456,18.67929498304404 ,0],
            [ 0.14588966182806892,4.087847921168613 ,0],
            [ 0.14776753752903887,6.049859155246893 ,0]
        ]

        return (thetas_500, thetas_300)

    # unconstrained limited search 
    def get_thetas_npl():
        thetas_500 =[
            [ 0.048143809993529844,9.436033461220461 ,0],
            [ 0.1056183804995852,5.392330379809875 ,0],
            [ 0.05047641810503022,14.740985595356078 ,0],
            [ 0.04345408837604258,14.619167963376459 ,0],
            [ 0.006425718212997706,58.96614306087356 ,0],
            [ 0.017939503018732036,25.33503714287109 ,0],
            [ 0.001230579965563759,413.3576093634338 ,0],
            [ 0.03426889630362318,13.460543838041371 ,0],
            [ 0.0014668532779062553,274.864947864665 ,0],
            [ 0.09812725630202107,4.659331166575832 ,0],
            [ 0.0015120703031258084,174.58025059484672 ,0],
            [ 0.05541976860783854,14.758358493103868 ,0],
            [ 0.05982582170512098,8.337819339943286 ,0],
            [ 0.0026369081163043224,114.6794726645872 ,0],
            [ 0.0008305787029152554,727.7393411056452 ,0],
            [ 0.011430810252772278,46.703579388977445 ,0],
            [ 0.05021271659888294,14.725378785127038 ,0],
            [ 0.04635224267520849,14.735229008660951 ,0],
            [ 0.006631542566164843,59.49362784160488 ,0],
            [ 0.005234661167585558,85.60779556128716 ,0],
            [ 0.004891451570117042,79.3730232301173 ,0],
            [ 0.05534941296841141,14.764659358814914 ,0]
        ]

        thetas_300 = [
            [ 0.10712050772783728,6.64248862770725 ,0],
            [ 0.007837993821896903,22.23165683086073 ,0],
            [ 0.06594482035086197,10.54594148952979 ,0],
            [ 0.08628633674262923,8.548101907519474 ,0],
            [ 0.08652529660511085,10.598369734867925 ,0],
            [ 0.035190303882175804,13.853310150856778 ,0],
            [ 0.012988197249833392,40.7883499176622 ,0],
            [ 0.1446487266632479,6.445272682211128 ,0],
            [ 0.05137091770542118,15.496575444989706 ,0],
            [ 0.043911829554060644,8.360995970181822 ,0],
            [ 0.017451896429040853,16.482462888055558 ,0],
            [ 0.12186520584935569,10.926249905830044 ,0],
            [ 0.038431902773315284,13.065077418709397 ,0],
            [ 0.014342160568855499,25.82711382503528 ,0],
            [ 0.02740881413928593,23.797897008791832 ,0],
            [ 0.08823061786334316,7.87424042125342 ,0],
            [ 0.027438025845009914,27.001410482259484 ,0],
            [ 0.08090476616791505,10.626001468685015 ,0],
            [ 0.06684684225760838,10.620488916446348 ,0],
            [ 0.024791679166429416,31.112162007729044 ,0],
            [ 0.1440284088798248,4.145140454984744 ,0],
            [ 0.14547436146035733,6.176397678151919 ,0]
        ]

        return (thetas_500, thetas_300)

def feature_index(feature,index):
    return int(index + (feature * 3) - 1)

def Feature_p_chosen(Theta, t, Choices, Outcomes, Stimuli, Relevant):
    eta, beta = Theta 

    W = np.zeros(9)
    penalty_sum = 0 
    previous_relevant_dim = -1
    for trail_num in range(len(Choices)):
        if(math.isnan(Choices[trail_num][0])):
            continue
        
        if(Relevant[trail_num] != previous_relevant_dim):
            W = np.zeros(9)

        # Calculate the probability that the user selected choice would be chosen:
        choice_value = 0
        for choice_index in range(0,3):
            choice_value += W[feature_index(choice_index,Choices[trail_num][choice_index])]
        denominator = 0
        for option_index in range(0,3):
            option_value = 0
            for option_feature in range(0,3):
                option_value += W[feature_index(option_feature,Stimuli[trail_num][option_index][option_feature])]
            try:
                denominator += math.exp(beta * option_value)
            except OverflowError: 
                denominator = float('inf')
        
        penalty = 0
        if(math.isfinite(denominator) and  denominator != 0):
            probability = math.exp(beta * choice_value) / denominator
            #print(probability)
            penalty += math.log(probability)
        else:
            penalty = float('inf')

        penalty_sum += penalty

        W[feature_index(0,Choices[trail_num][0])] += (eta * (Outcomes[trail_num] - choice_value))
        W[feature_index(1,Choices[trail_num][1])] += (eta * (Outcomes[trail_num] - choice_value))
        W[feature_index(2,Choices[trail_num][2])] += (eta * (Outcomes[trail_num] - choice_value))

        previous_relevant_dim = Relevant[trail_num]

    # check if log pdf is 0 
    log_prior = 0
    
    
    if (beta == 0):
        log_prior = np.inf
    else: 
        log_prior = scipy.stats.gamma.logpdf(beta, 2, loc=0, scale=3)
    
    return -1 * (penalty_sum + log_prior)

def Feature_p_chosen_delta(Theta, t, Choices, Outcomes, Stimuli, Relevant):
    eta, beta, delta = Theta 

    W = np.zeros(9)
    penalty_sum = 0 
    previous_relevant_dim = -1
    for trail_num in range(len(Choices)):
        if(math.isnan(Choices[trail_num][0])):
            continue
        
        if(Relevant[trail_num] != previous_relevant_dim):
            W = np.zeros(9)

        # Calculate the probability that the user selected choice would be chosen:
        choice_value = 0
        for choice_index in range(0,3):
            choice_value += W[feature_index(choice_index,Choices[trail_num][choice_index])]
        denominator = 0
        for option_index in range(0,3):
            option_value = 0
            for option_feature in range(0,3):
                option_value += W[feature_index(option_feature,Stimuli[trail_num][option_index][option_feature])]
            try:
                denominator += math.exp(beta * option_value)
            except OverflowError: 
                denominator = float('inf')
        
        penalty = 0
        if(math.isfinite(denominator) and  denominator != 0):
            penalty = beta * choice_value
            penalty -= math.log(denominator)
        else:
            penalty = -1 * float('inf')

        penalty_sum += penalty
        #print(W)

        W[feature_index(0,Choices[trail_num][0])] += (eta * (Outcomes[trail_num] - choice_value))
        W[feature_index(1,Choices[trail_num][1])] += (eta * (Outcomes[trail_num] - choice_value))
        W[feature_index(2,Choices[trail_num][2])] += (eta * (Outcomes[trail_num] - choice_value))

        #W[feature_index(0,Choices[trail_num][0])] += (eta * (Outcomes[trail_num] - W[feature_index(0,Choices[trail_num][0])]))
        #W[feature_index(1,Choices[trail_num][1])] += (eta * (Outcomes[trail_num] - W[feature_index(1,Choices[trail_num][1])]))
        #W[feature_index(2,Choices[trail_num][2])] += (eta * (Outcomes[trail_num] - W[feature_index(2,Choices[trail_num][2])]))
        
        #forget
        chosen_features = [feature_index(0,Choices[trail_num][0]), feature_index(1,Choices[trail_num][1]), feature_index(2,Choices[trail_num][2])]
        for w_feature_index in range(0,len(W)):
            if(not w_feature_index in chosen_features):
                W[w_feature_index] *= (1 - delta)

        previous_relevant_dim = Relevant[trail_num]

    # check if log pdf is 0 
    log_prior = 0
    if (beta == 0):
        log_prior = -1 * np.inf
    else: 
        log_prior = scipy.stats.gamma.logpdf(beta, 2, loc=0, scale=3)
    return -1 * (penalty_sum + log_prior)

def print_NLL():

    mat = scipy.io.loadmat('BehavioralDataOnline.mat')

    Choices = mat['DimTaskData'][0][0]['Choices']
    Outcomes = mat['DimTaskData'][0][0]['Outcomes']
    Stimuli = mat['DimTaskData'][0][0]['Stimuli']
    RelevantDim = mat['DimTaskData'][0][0]['RelevantDim']
    CorrectFeature = mat['DimTaskData'][0][0]['CorrectFeature']
    ReactionTimes = mat['DimTaskData'][0][0]['ReactionTimes']

    thetas_300 =[
        [ 0.11121441964878809,6.398816971302837 ,0],
        [ 0.03626052377496113,5.608728563437298 ,0],
        [ 0.06626244857349133,10.51702482175613 ,0],
        [ 0.09881663534687445,7.673210034445709 ,0],
        [ 0.0865394456537615,10.590332576210095 ,0],
        [ 0.04665730887340656,10.66401260454219 ,0],
        [ 0.024651286980226634,21.746369306539737 ,0],
        [ 0.14567174984356973,6.376588500024989 ,0],
        [ 0.07458720549158034,10.62615779451802 ,0],
        [ 0.04504873849924031,8.202211167714111 ,0],
        [ 0.018663424773087377,15.456177781668528 ,0],
        [ 0.12202141721517153,10.89745585470039 ,0],
        [ 0.043067200622488234,11.84124368512711 ,0],
        [ 0.036126668083147434,10.97243972592328 ,0],
        [ 0.04104078824187614,16.12668587209663 ,0],
        [ 0.09629718461779581,7.201514637283343 ,0],
        [ 0.07008113131345609,10.648046706851204 ,0],
        [ 0.08090986908020911,10.623203402080138 ,0],
        [ 0.06711638678481537,10.59349319502355 ,0],
        [ 0.04449257409288976,18.212811188918916 ,0],
        [ 0.11481943882762802,5.20218115164325 ,0],
        [ 0.14777004178773076,6.048703278346842 ,0]
    ]

    thetas_500 = [
        [ 0.05247199196384004,8.765698220607128 ,0],
        [ 0.0342562745967956,12.114503558136889 ,0],
        [ 0.050524919757222064,14.738409929166192 ,0],
        [ 0.0707719027505329,9.864032130698671 ,0],
        [ 0.027252182152597063,15.51934820548859 ,0],
        [ 0.03564855607809842,13.853817943034418 ,0],
        [ 0.021803226095470005,25.266793119211776 ,0],
        [ 0.043888395456981436,10.901282853822122 ,0],
        [ 0.021463979000174875,20.101755701131353 ,0],
        [ 0.10007805660276671,4.590984473833376 ,0],
        [ 0.01977261748751931,14.317930357561691 ,0],
        [ 0.0554279412241022,14.753371607140844 ,0],
        [ 0.0662110047783497,7.657342653183934 ,0],
        [ 0.029038385852828953,11.687674980348442 ,0],
        [ 0.041081586576455165,14.782079725162502 ,0],
        [ 0.031007736038584404,18.386870375777605 ,0],
        [ 0.050256395377604374,14.723314723526377 ,0],
        [ 0.04642485257270639,14.730980614661986 ,0],
        [ 0.02756145791009094,16.228747588552203 ,0],
        [ 0.025654147166787326,19.24196862012504 ,0],
        [ 0.03289205872456531,13.341154766016725 ,0],
        [ 0.055340365456880806,14.759195825219114 ,0]
    ]

    bounds = Bounds([0, 0], [0.5, np.inf])
    #bounds_delta = Bounds([0, 0, 0], [1, np.inf, 1])
    bounds_delta = Bounds([0, 0, 0], [1, np.inf, 1])

    NLLs = []

    for participant_id in range(0,22):
        choices_500 = Choices[0:500,0:3,participant_id]
        outcomes_500 = Outcomes[0:500,participant_id]
        stimuli_500 = Stimuli[0:500,0:3,0:3,participant_id]
        relevant_500 = RelevantDim[0:500,participant_id]

        choices_300 = Choices[500:800,0:3,participant_id]
        outcomes_300 = Outcomes[500:800,participant_id]
        stimuli_300 = Stimuli[500:800,0:3,0:3,participant_id]
        relevant_300 = RelevantDim[500:800,participant_id]

        Theta = [0.02, 20, 0]

        NLLs.append(Feature_p_chosen(Theta, 500, choices_500, outcomes_500, stimuli_500, relevant_500)) 
        NLLs.append(Feature_p_chosen(Theta, 300, choices_300, outcomes_300, stimuli_300, relevant_300))

        #NLLs.append(Feature_p_chosen(thetas_500[participant_id], 500, choices_500, outcomes_500, stimuli_500, relevant_500))
        #NLLs.append(Feature_p_chosen(thetas_300[participant_id], 300, choices_300, outcomes_300, stimuli_300, relevant_300))
    
    print(np.mean(NLLs))


#print_NLL()

def calculate_Thetas():

    mat = scipy.io.loadmat('BehavioralDataOnline.mat')

    Choices = mat['DimTaskData'][0][0]['Choices']
    Outcomes = mat['DimTaskData'][0][0]['Outcomes']
    Stimuli = mat['DimTaskData'][0][0]['Stimuli']
    RelevantDim = mat['DimTaskData'][0][0]['RelevantDim']
    CorrectFeature = mat['DimTaskData'][0][0]['CorrectFeature']
    ReactionTimes = mat['DimTaskData'][0][0]['ReactionTimes']

    thetas_500 = []
    thetas_300 = []
    thetas_800 = []
    bounds = Bounds([0, 0], [0.5, np.inf])
    #bounds_delta = Bounds([0, 0, 0], [1, np.inf, 1])
    bounds_delta = Bounds([0, 0, 0], [1, np.inf, 1])

    NLL_SUM = 0 

    for participant_id in range(0,22):
        choices_500 = Choices[0:500,0:3,participant_id]
        outcomes_500 = Outcomes[0:500,participant_id]
        stimuli_500 = Stimuli[0:500,0:3,0:3,participant_id]
        relevant_500 = RelevantDim[0:500,participant_id]

        choices_300 = Choices[500:800,0:3,participant_id]
        outcomes_300 = Outcomes[500:800,participant_id]
        stimuli_300 = Stimuli[500:800,0:3,0:3,participant_id]
        relevant_300 = RelevantDim[500:800,participant_id]

        choices_800 = Choices[0:800,0:3,participant_id]
        outcomes_800 = Outcomes[0:800,participant_id]
        stimuli_800 = Stimuli[0:800,0:3,0:3,participant_id]
        relevant_800 = RelevantDim[0:800,participant_id]

        initial_guess_500 = [0.047, 14.73]
        initial_guess_300 = [0.076, 10.62]
        initial_guess_800 = [0.055, 12]

        initial_guess_500_delta = [0.122, 0.466, 10.33]
        initial_guess_300_delta = [0.151, 0.420, 9.18]

        feature_fitted_params_500 = [-1, -1, -1]
        feature_fitted_params_300 = [-1, -1, -1]

        
        """feature_result_300 = optimize.minimize(Feature_p_chosen, initial_guess_300, 
                                        args=(300, choices_300, outcomes_300, stimuli_300, relevant_300), 
                                        method='L-BFGS-B',
                                        #options={'maxiter':10000, 'ftol':0.00001, 'maxcor':1},
                                        bounds=bounds)
        
        if feature_result_300.success:
            feature_fitted_params_300 = feature_result_300.x
            #print("[",','.join(map(str, feature_fitted_params_300)), "]," ) 
            print("[",','.join(map(str, feature_fitted_params_300)),",0],") 
        else:
            print(feature_result_300.message)
        thetas_300.append(feature_fitted_params_300)"""

        

        feature_result_500 = optimize.minimize(Feature_p_chosen, initial_guess_500, 
                                        args=(500, choices_500, outcomes_500, stimuli_500, relevant_500), 
                                        method='L-BFGS-B',
                                        #options={'maxiter':10000, 'ftol':0.00001},
                                        bounds=bounds)


        if feature_result_500.success:
            feature_fitted_params_500 = feature_result_500.x
            #print("[",','.join(map(str, feature_fitted_params_500)),"],") 
            print("[",','.join(map(str, feature_fitted_params_500)),",0],") 

        else:
            print(feature_result_500.message)
        thetas_500.append(feature_fitted_params_500)
        

        """feature_result_800 = optimize.minimize(Feature_p_chosen, initial_guess_800, 
                                        args=(800, choices_800, outcomes_800, stimuli_800, relevant_800), 
                                        method='L-BFGS-B',
                                        options={'maxiter':10000, 'ftol':0.00001},
                                        bounds=bounds)


        if feature_result_800.success:
            feature_fitted_params_800 = feature_result_800.x
            #print("[",','.join(map(str, feature_fitted_params_500)),"],") 
            print("[",','.join(map(str, feature_fitted_params_800)),",0],") 

        else:
            print(feature_result_800.message)
        thetas_800.append(feature_fitted_params_800)"""
        
        
    return (thetas_500, thetas_300, thetas_800)

def calculate_NLL():

    mat = scipy.io.loadmat('BehavioralDataOnline.mat')

    Choices = mat['DimTaskData'][0][0]['Choices']
    Outcomes = mat['DimTaskData'][0][0]['Outcomes']
    Stimuli = mat['DimTaskData'][0][0]['Stimuli']
    RelevantDim = mat['DimTaskData'][0][0]['RelevantDim']
    CorrectFeature = mat['DimTaskData'][0][0]['CorrectFeature']
    ReactionTimes = mat['DimTaskData'][0][0]['ReactionTimes']

    thetas_500 =[
            [ 0.048143809993529844,9.436033461220461],
            [ 0.1056183804995852,5.392330379809875],
            [ 0.05047641810503022,14.740985595356078],
            [ 0.04345408837604258,14.619167963376459],
            [ 0.006425718212997706,58.96614306087356],
            [ 0.017939503018732036,25.33503714287109],
            [ 0.001230579965563759,413.3576093634338],
            [ 0.03426889630362318,13.460543838041371],
            [ 0.0014668532779062553,274.864947864665],
            [ 0.09812725630202107,4.659331166575832],
            [ 0.0015120703031258084,174.58025059484672],
            [ 0.05541976860783854,14.758358493103868],
            [ 0.05982582170512098,8.337819339943286],
            [ 0.0026369081163043224,114.6794726645872],
            [ 0.0008305787029152554,727.7393411056452],
            [ 0.011430810252772278,46.703579388977445],
            [ 0.05021271659888294,14.725378785127038],
            [ 0.04635224267520849,14.735229008660951],
            [ 0.006631542566164843,59.49362784160488],
            [ 0.005234661167585558,85.60779556128716],
            [ 0.004891451570117042,79.3730232301173],
            [ 0.05534941296841141,14.764659358814914]
        ]

    thetas_300 = [
            [ 0.10712050772783728,6.64248862770725],
            [ 0.007837993821896903,22.23165683086073],
            [ 0.06594482035086197,10.54594148952979],
            [ 0.08628633674262923,8.548101907519474],
            [ 0.08652529660511085,10.598369734867925],
            [ 0.035190303882175804,13.853310150856778],
            [ 0.012988197249833392,40.7883499176622],
            [ 0.1446487266632479,6.445272682211128],
            [ 0.05137091770542118,15.496575444989706],
            [ 0.043911829554060644,8.360995970181822],
            [ 0.017451896429040853,16.482462888055558],
            [ 0.12186520584935569,10.926249905830044],
            [ 0.038431902773315284,13.065077418709397],
            [ 0.014342160568855499,25.82711382503528],
            [ 0.02740881413928593,23.797897008791832],
            [ 0.08823061786334316,7.87424042125342],
            [ 0.027438025845009914,27.001410482259484],
            [ 0.08090476616791505,10.626001468685015],
            [ 0.06684684225760838,10.620488916446348],
            [ 0.024791679166429416,31.112162007729044],
            [ 0.1440284088798248,4.145140454984744],
            [ 0.14547436146035733,6.176397678151919]
        ]


    NLL_SUM = 0 

    for participant_id in range(0,22):
        choices_500 = Choices[0:500,0:3,participant_id]
        outcomes_500 = Outcomes[0:500,participant_id]
        stimuli_500 = Stimuli[0:500,0:3,0:3,participant_id]
        relevant_500 = RelevantDim[0:500,participant_id]

        choices_300 = Choices[500:800,0:3,participant_id]
        outcomes_300 = Outcomes[500:800,participant_id]
        stimuli_300 = Stimuli[500:800,0:3,0:3,participant_id]
        relevant_300 = RelevantDim[500:800,participant_id]
        
        initial_guess_500 = [0.047, 14.73]
        initial_guess_300 = [0.076, 10.62]

        initial_guess_500_delta = [0.122, 0.466, 10.33]
        initial_guess_300_delta = [0.151, 0.420, 9.18]

        feature_fitted_params_500 = [-1, -1, -1]
        feature_fitted_params_300 = [-1, -1, -1]

        NLL_SUM += Feature_p_chosen(thetas_500[participant_id], 500, choices_500, outcomes_500, stimuli_500, relevant_500)
        NLL_SUM += Feature_p_chosen(thetas_300[participant_id], 300, choices_300, outcomes_300, stimuli_300, relevant_300)

    return NLL_SUM
def Information_Rate(Theta, t, Choices, Outcomes, Stimuli, Relevant):
    eta, beta = Theta 

    #print(Theta)

    W = np.zeros(9)
    penalty_sum = 0 
    previous_relevant_dim = -1
    entropy_sum = 0
    for trail_num in range(len(Choices)):
        if(math.isnan(Choices[trail_num][0])):
            continue
        
        if(Relevant[trail_num] != previous_relevant_dim):
            W = np.zeros(9)

        # Calculate the probability that the user selected choice would be chosen:
        choice_value = 0
        for choice_index in range(0,3):
            choice_value += W[feature_index(choice_index,Choices[trail_num][choice_index])]
        denominator = 0
        for option_index in range(0,3):
            option_value = 0
            for option_feature in range(0,3):
                option_value += W[feature_index(option_feature,Stimuli[trail_num][option_index][option_feature])]
            try:
                denominator += math.exp(beta * option_value)
            except OverflowError: 
                denominator = float('inf')
        
        probabilities = [1/3,1/3,1/3]

        for option_index in range(0,3):
            option_value = 0
            for option_feature in range(0,3):
                option_value += W[feature_index(option_feature,Stimuli[trail_num][option_index][option_feature])]
            if(math.isfinite(denominator) and  denominator != 0):
                probabilities[option_index] = math.exp((beta * option_value)) / denominator

        
        entropy_sum += scipy.stats.entropy([1/3,1/3,1/3],probabilities)

        W[feature_index(0,Choices[trail_num][0])] += (eta * (Outcomes[trail_num] - choice_value))
        W[feature_index(1,Choices[trail_num][1])] += (eta * (Outcomes[trail_num] - choice_value))
        W[feature_index(2,Choices[trail_num][2])] += (eta * (Outcomes[trail_num] - choice_value))

        previous_relevant_dim = Relevant[trail_num]

    # check if log pdf is 0 
    log_prior = 0
    return entropy_sum
def calculate_Information():
        mat = scipy.io.loadmat('BehavioralDataOnline.mat')

        Choices = mat['DimTaskData'][0][0]['Choices']
        Outcomes = mat['DimTaskData'][0][0]['Outcomes']
        Stimuli = mat['DimTaskData'][0][0]['Stimuli']
        RelevantDim = mat['DimTaskData'][0][0]['RelevantDim']
        CorrectFeature = mat['DimTaskData'][0][0]['CorrectFeature']
        ReactionTimes = mat['DimTaskData'][0][0]['ReactionTimes']

        thetas_500 =[
            [ 0.048143809993529844,9.436033461220461],
            [ 0.1056183804995852,5.392330379809875],
            [ 0.05047641810503022,14.740985595356078],
            [ 0.04345408837604258,14.619167963376459],
            [ 0.006425718212997706,58.96614306087356],
            [ 0.017939503018732036,25.33503714287109],
            [ 0.001230579965563759,413.3576093634338],
            [ 0.03426889630362318,13.460543838041371],
            [ 0.0014668532779062553,274.864947864665],
            [ 0.09812725630202107,4.659331166575832],
            [ 0.0015120703031258084,174.58025059484672],
            [ 0.05541976860783854,14.758358493103868],
            [ 0.05982582170512098,8.337819339943286],
            [ 0.0026369081163043224,114.6794726645872],
            [ 0.0008305787029152554,727.7393411056452],
            [ 0.011430810252772278,46.703579388977445],
            [ 0.05021271659888294,14.725378785127038],
            [ 0.04635224267520849,14.735229008660951],
            [ 0.006631542566164843,59.49362784160488],
            [ 0.005234661167585558,85.60779556128716],
            [ 0.004891451570117042,79.3730232301173],
            [ 0.05534941296841141,14.764659358814914]
        ]

        thetas_300 = [
            [ 0.10712050772783728,6.64248862770725],
            [ 0.007837993821896903,22.23165683086073],
            [ 0.06594482035086197,10.54594148952979],
            [ 0.08628633674262923,8.548101907519474],
            [ 0.08652529660511085,10.598369734867925],
            [ 0.035190303882175804,13.853310150856778],
            [ 0.012988197249833392,40.7883499176622],
            [ 0.1446487266632479,6.445272682211128],
            [ 0.05137091770542118,15.496575444989706],
            [ 0.043911829554060644,8.360995970181822],
            [ 0.017451896429040853,16.482462888055558],
            [ 0.12186520584935569,10.926249905830044],
            [ 0.038431902773315284,13.065077418709397],
            [ 0.014342160568855499,25.82711382503528],
            [ 0.02740881413928593,23.797897008791832],
            [ 0.08823061786334316,7.87424042125342],
            [ 0.027438025845009914,27.001410482259484],
            [ 0.08090476616791505,10.626001468685015],
            [ 0.06684684225760838,10.620488916446348],
            [ 0.024791679166429416,31.112162007729044],
            [ 0.1440284088798248,4.145140454984744],
            [ 0.14547436146035733,6.176397678151919]
        ]

        NLL_SUM = 0

        #thetas_500 = []
        #thetas_300 = []
        entropy_sum = 0
        for participant_id in range(0,22):
            choices_500 = Choices[0:500,0:3,participant_id]
            outcomes_500 = Outcomes[0:500,participant_id]
            stimuli_500 = Stimuli[0:500,0:3,0:3,participant_id]
            relevant_500 = RelevantDim[0:500,participant_id]

            choices_300 = Choices[500:800,0:3,participant_id]
            outcomes_300 = Outcomes[500:800,participant_id]
            stimuli_300 = Stimuli[500:800,0:3,0:3,participant_id]
            relevant_300 = RelevantDim[500:800,participant_id]

            #eta, delta, beta, 
            #bounds = Bounds([0, 0, 0], [1, 1, np.inf])
            bounds = Bounds([0, 0, 0], [1, 0, np.inf])


            entropy_sum += Information_Rate(thetas_500[participant_id], 500, choices_500, outcomes_500, stimuli_500, relevant_500)
            entropy_sum += Information_Rate(thetas_300[participant_id], 300, choices_300, outcomes_300, stimuli_300, relevant_300)
            
        return entropy_sum

