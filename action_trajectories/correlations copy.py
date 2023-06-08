import pickle5 as pickle
import sys
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from minepy import MINE
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
numpy.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os
import re
greedy0 = "action_trajectories/greedy_and_greedy_agent0.p"
greedy1 = "action_trajectories/greedy_and_greedy_agent1.p"
idle0 = "action_trajectories/greedy_and_idle_agent0.p"
idle1 = "action_trajectories/greedy_and_idle_agent1.p"
untrained0 = "action_trajectories/untrained_and_untrained_agent0-agent1.p"
untrained1 = "action_trajectories/untrained_and_untrained_agent1-agent0.p"

# piston ball histories
pb_untrained = "pistonball/actions_hist_untrained.pkl"
pb_trained_10001 = "pistonball/actions_hist_trained10001.pkl"
pb_trained25 = "pistonball/trained_25.pkl"
pb_untrained25 = "pistonball/untrained_25.pkl"
pb_trained100 = "pistonball/trained_100.pkl"
pb_untrained100 = "pistonball/untrained_100.pkl"
pb_trained100mic0 = "pistonball/trained_100mic.pkl"
pb_trained100mic = "pistonball/trained10001_mic100.pkl"


def get_overcooked_MIC(file_path):

    with open(file_path, 'rb') as file:
        trajectory_info = pickle.load(file)
 

    # print(trajectory_info.keys())
    # Print the loaded dictionary
    actions_dic = {}
    actions_dic = {(0, 0): 0, (0, -1): 1, (-1, 0): 2, 'interact': 3, (1, 0): 4, (0, 1): 5}
    action_idx = 0
    actions_list = [[],[]]
    for _ , actions in enumerate(trajectory_info['ep_actions'].squeeze(0)):
        # print(f"{actions}")
        for player, action in enumerate(actions):
            # if action not in actions_dic:
            #     actions_dic[action] = action_idx
            #     action_idx +=1
            actions_list[player].append(actions_dic[action])
    player1, player2 = actions_list
    actions_list = np.array([player1, player2])
    start = 0
    finish = 1001
    # print(actions_dic)
    mine = MINE(alpha=.6, c=15, est="mic_approx")
    mine.compute_score(player1[start:finish], player2[start:finish])
    
    kendalltau = stats.kendalltau(player1, player2)[0]
    pearson = stats.pearsonr(player1, player2)[0]
    spearmanr = stats.spearmanr(player1, player2)[0]
    # # print(kendalltau)
    # mine.compute_score([1,2,3], [2,4,6])
    return (mine.mic())#, kendalltau, pearson, spearmanr)    
    # print(episode[actor])
    
        
def getOK_MICstats():
    greedy0 = "action_trajectories/greedy_and_greedy_agent0.p"
    greedy1 = "action_trajectories/greedy_and_greedy_agent1.p"
    idle0 = "action_trajectories/greedy_and_idle_agent0.p"
    idle1 = "action_trajectories/greedy_and_idle_agent1.p"
    untrained0 = "action_trajectories/untrained_and_untrained_agent0-agent1.p"
    untrained1 = "action_trajectories/untrained_and_untrained_agent1-agent0.p"

    idle0, idle1 = (get_overcooked_MIC(idle0), get_overcooked_MIC(idle1))
    untrained0, untrained1 = (get_overcooked_MIC(untrained0), get_overcooked_MIC(untrained1))
    greedy0, greedy1 = (get_overcooked_MIC(greedy0), get_overcooked_MIC(greedy1))

    print(idle0, idle0)
    print(untrained0, untrained1)
    print(greedy0, greedy1)
    behaviors = np.array([[untrained1, untrained0],
                        [idle1, idle0],
                        [greedy0, greedy1]]
                        )
    # print(behaviors[:,0])

    behavior_mine = MINE(alpha=.6, c=15, est="mic_approx")
    behavior_mine.compute_score(behaviors[:,0], behaviors[:,1])
    print(behavior_mine.mic())
    
    
getOK_MICstats()
    
def get_pistonball_MIC(file_path):

    with open(file_path, 'rb') as file:
        trajectory_info = pickle.load(file)
 

    # print(trajectory_info.keys())
    ep_no = 3
    episode = trajectory_info[ep_no]    
    actor = 0
    # for actor in range(episode.shape[0]): # comment out for single case
    mine = MINE(alpha=.6, c=15, est="mic_approx")
    mine.compute_score(episode[actor].cpu(), episode[actor].cpu())
    # print(episode[actor].cpu())
    # print(episode[actor].cpu())
    print(f"{mine.mic()}")
    action_hist = {}
    for partner in range(episode.shape[0]):
        mine.compute_score(episode[actor].cpu(), episode[partner].cpu())
        print(f"({partner}: {mine.mic()}")
        action_hist[partner] = mine.mic()
    x = list(action_hist.keys())
    y = list(action_hist.values())
    
    x_smooth = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y, k=1)
    y_smooth = spl(x_smooth)
    plt.plot(x, y, linestyle='-', linewidth=1, color='black', alpha=0.6, zorder=0)  
    # plt.plot(x, y, linestyle='-', color='black', alpha=0.6)
    
    plt.xlabel('Agent ID #')
    plt.ylabel('MIC Value')
    plt.title(f'MIC Values for Agent {actor} and Partners')

    plt.bar(x, y, s=10)
    ### indent above for multi-case
    plt.xlim(-1, 20)
    plt.xticks(range(20))

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')

# Show the plot
# get_pistonball_MIC(pb_untrained)
# get_pistonball_MIC(pb_trained_10001)

# plt.show()

def normalized_entropy(x, y):
    data = x + y
    unique, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    max_entropy = -(probs * np.log2(probs)).sum()
    norm_entropy = 0.0 if max_entropy == 0 else (max_entropy / np.log2(len(unique)))
    return norm_entropy

def get_pistonball_MIC(file_path, actor, ep_no):

    with open(file_path, 'rb') as file:
        trajectory_info = pickle.load(file)

    legend_label = os.path.splitext(os.path.basename(file_path))[0].split('last_')[-1]

    episode = trajectory_info[ep_no]
    mine = MINE(alpha=.6, c=15, est="mic_approx")
    mine.compute_score(episode[actor].cpu(), episode[actor].cpu())
    print(f"{mine.mic()}")
    action_hist = {}
    for partner in range(episode.shape[0]):
        mine.compute_score(episode[actor].cpu(), episode[partner].cpu())
        print(f"({partner}: {mine.mic()}")
        
        action_hist[partner] = mine.mic()
    x = list(action_hist.keys())
    y = list(action_hist.values())

    x_smooth = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y, k=1)
    y_smooth = spl(x_smooth)
    plt.plot(x_smooth, y_smooth, linestyle='-', linewidth=1, color='black', alpha=0.5, zorder=0)

    plt.xlabel('Agent ID #')
    plt.ylabel('MIC Value')
    plt.title(f'MIC Values for Agent {actor} and Partners')

    plt.scatter(x, y, s=10, label=legend_label)
    plt.xlim(20, -1)  
    plt.xticks(range(20), range(1, 21))  

    plt.xlabel('Agent ID #')
    plt.ylabel('MIC')
    plt.title('Scatter Plot')
    
class ActorBarChart:
    def __init__(self):
        self.total_actors = 0
        self.actors = []
        self.plot_data = []
        self.stat = None

    def add_actor(self, file_path, actor, stat="mic"):
        self.total_actors += 1
        self.actors.append(actor)
        self.stat = stat

        print(stat)
        with open(file_path, 'rb') as file:
            trajectory_info = pickle.load(file)

        legend_label = os.path.splitext(os.path.basename(file_path))[0].split('last_')[-1]

        action_hist = {}
        
        num_episodes = len(trajectory_info)
        for ep in range(num_episodes):
            episode = trajectory_info[ep]
            for partner in range(episode.shape[0]):
                if stat is "mic":
                    mine = MINE(alpha=.6, c=15, est="mic_approx")
                    mine.compute_score(episode[actor].cpu(), episode[partner].cpu())
                    score = mine.gmic()
                elif stat is "kendall":
                    score = stats.kendalltau(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat is "pearson":
                    score = stats.pearsonr(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat is "spearman":
                    score = stats.spearmanr(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat is "norm_entropy":
                    score = normalized_entropy(episode[actor].cpu(), episode[partner].cpu())
                else:
                    raise("Not a valid stat type")

                if partner not in action_hist:
                    action_hist[partner] = 0
                action_hist[partner] += score/ num_episodes

        x = list(action_hist.keys())
        y = list(action_hist.values())

        # Calculate bar width and offset for any number of actors

        self.plot_data.append({"actor": actor, "x": np.array(x), "y": y, "dataset":legend_label})
        
        

    def plot(self):
        if self.stat is None:
            raise ValueError("No actors have been added for plotting.")

        bar_width = 1.0 / (self.total_actors + 1)

        for i, data in enumerate(self.plot_data):
            actor = data["actor"]
            dataset = data["dataset"]
            offset = (i - 1) * bar_width
            
            plt.bar(data["x"] + offset, data["y"], width=bar_width, label=f"Actor {actor}", zorder=2)
            # plt.bar(data["x"] + offset, data["y"], width=bar_width, label=f"{dataset} {self.stat}", zorder=2)

        plt.xlim(20, -1)
        plt.xticks(range(20))
        plt.yticks(np.arange(0, 1.2, .1))

        plt.xlabel('Partner ID #')
        plt.ylabel(f'{self.stat} Value')
        plt.title(f'{self.stat} Values for Multiple Actors')

        plt.legend(loc='best')
        plt.grid(alpha=.3, zorder=0)
        plt.show()

        
    def show(self):
        plt.legend()
        plt.grid(alpha=.3, zorder=0)
        
# bar_chart = ActorBarChart()
# for actor in range(5,10):
#     bar_chart.add_actor(pb_trained100mic, actor, "mic")
# bar_chart.plot()
# plt.show()


class ActorPartnerDistance:
    def __init__(self):
        self.total_actors = 0
        self.distance_hist = []

    def add_actor(self, file_path, actor, stat="mic"):
        self.total_actors += 1

        with open(file_path, 'rb') as file:
            trajectory_info = pickle.load(file)
        num_agents = trajectory_info[0].shape[0]

        num_distances = num_agents - 1
        action_hist = np.zeros(num_distances)

        num_episodes = len(trajectory_info)
        for ep in range(num_episodes):
            episode = trajectory_info[ep]
            for partner in range(episode.shape[0]):
                if actor == partner:
                    continue
                distance = abs(actor - partner) - 1

                if stat == "mic":
                    mine = MINE(alpha=.6, c=15, est="mic_approx")
                    mine.compute_score(episode[actor].cpu(), episode[partner].cpu())
                    score = mine.mic()
                elif stat == "kendall":
                    score = stats.kendalltau(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat == "pearson":
                    score = stats.pearsonr(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat == "spearman":
                    score = stats.spearmanr(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat == "norm_entropy":
                    score = normalized_entropy(episode[actor].cpu(), episode[partner].cpu())
                else:
                    raise ValueError("Not a valid stat type")

                action_hist[distance] += score / num_episodes

        self.distance_hist.append(action_hist)

    def plot(self):
        if self.total_actors == 0:
            raise ValueError("No actors have been added for plotting.")

        distance_hist = np.array(self.distance_hist)
        mu = np.mean(distance_hist, axis=0)
        upper = mu + np.std(distance_hist, axis=0)
        lower = mu - np.std(distance_hist, axis=0)

        x = np.arange(1, distance_hist.shape[1] + 1)

        plt.fill_between(x, upper, lower, alpha=0.2)
        plt.plot(x, mu, marker="o", alpha=0.45)

        plt.xlabel("Relative Partner Distance")
        plt.ylabel("MIC")
        plt.xlim(0,20)  
        plt.xticks(range(20))  
        # plt.xticks(range(19))
        plt.title("MIC by Relative Partner Distance for All Agents")
        plt.grid(alpha=.3, zorder=0)

    def show(self):
        plt.legend()
        plt.grid(alpha=.3, zorder=0)

# scatter_plot = ActorPartnerDistance()
# for actor in range(20):
#     scatter_plot.add_actor(pb_untrained100, actor, "mic")

# scatter_plot.plot()

        
# # bar_chart = ActorBarChart()
# bar_chart.add_actor(pb_trained100, 4, "mic")
# bar_chart.add_actor(pb_trained100, 15, "mic")
# # bar_chart.plot()
# plt.show()

class ActorPartnerDistance2:
    def __init__(self):
        self.total_actors = 0
        self.distance_hist = []

    def add_actor(self, file_path, actor, stat="mic"):
        self.total_actors += 1

        with open(file_path, 'rb') as file:
            trajectory_info = pickle.load(file)
        num_agents = trajectory_info[0].shape[0]

        num_distances = 18
        action_hist = np.zeros(num_distances)

        num_episodes = len(trajectory_info)
        for ep in range(num_episodes):
            episode = trajectory_info[ep]
            for partner in range(episode.shape[0]):
                if actor == partner:
                    continue
                distance = abs(actor - partner) - 1

                if stat == "mic":
                    mine = MINE(alpha=.6, c=15, est="mic_approx")
                    mine.compute_score(episode[actor].cpu(), episode[partner].cpu())
                    score = mine.mic()
                elif stat == "kendall":
                    score = stats.kendalltau(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat == "pearson":
                    score = stats.pearsonr(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat == "spearman":
                    score = stats.spearmanr(episode[actor].cpu(), episode[partner].cpu())[0]
                elif stat == "norm_entropy":
                    score = normalized_entropy(episode[actor].cpu(), episode[partner].cpu())
                else:
                    raise ValueError("Not a valid stat type")

                action_hist[distance] += score / num_episodes

        self.distance_hist.append(action_hist)

    def plot(self):
        if self.total_actors == 0:
            raise ValueError("No actors have been added for plotting.")

        distance_hist = np.array(self.distance_hist)
        mu = np.mean(distance_hist, axis=0)
        upper = mu + np.std(distance_hist, axis=0)
        lower = mu - np.std(distance_hist, axis=0)

        x = np.arange(1, distance_hist.shape[1] + 1)

        plt.fill_between(x, upper, lower, alpha=0.2)
        plt.plot(x, mu, marker="o", alpha=0.45)

        plt.xlabel("Relative Partner Distance")
        plt.ylabel("MIC")
        plt.title("MIC by Relative Partner Distance for All Agents")
        plt.grid(alpha=.3, zorder=0)

    def show(self):
        plt.legend()
        plt.grid(alpha=.3, zorder=0)

scatter_plot = ActorPartnerDistance()
scatter_plot2 = ActorPartnerDistance()
for actor in range(1,2):
    scatter_plot.add_actor(pb_trained100, actor, "mic")
    # scatter_plot2.add_actor(pb_trained100, actor, "mic")

scatter_plot.plot()
# scatter_plot2.plot()
plt.show()
# plt.show()
    
# actor = 1
# get_pistonball_MIC_avg(pb_untrained100, 0, "mic")
# get_pistonball_MIC_bar(pb_trained100, 4, "mic")
# get_pistonball_MIC_avg(pb_untrained100, 10, "mic")
# get_pistonball_MIC_avg(pb_trained100, 10, "mic")
# get_pistonball_MIC_bar(pb_trained100, 15, "mic")
# # get_pistonball_MIC_avg(pb_trained25, 8, "mic")
# get_pistonball_MIC_avg(pb_trained100, actor, "mic")
# # get_pistonball_MIC_avg(pb_trained25, 10, "mic")

# bar_chart = ActorPartnerDistance2()
# for actor in range(10,15):
#     bar_chart.add_actor(pb_trained100, actor, "mic")
# bar_chart.plot()
# plt.show()

    
# plt.legend()
# plt.grid(alpha = .3, zorder=0)
# plt.show()

def get_all_graph(stat):
    print(stat)
    
    if not os.path.exists(f"charts/{stat}"):
        os.makedirs(f"charts/{stat}")
    for actor in range(20):
        get_pistonball_MIC_avg(pb_untrained100, actor, stat)
        get_pistonball_MIC_avg(pb_trained100, actor, stat)    
        plt.legend()
        plt.grid(alpha = .3, zorder=0)
        
        # Save the plot with high resolution
        # plt.savefig(f"charts/{stat}_alt/actor_{actor}_{stat}.png", dpi=300)
        
        # Show the plot (optional, remove this line if you don't want to display the plots)
        plt.show()

        # Clear the current plot before starting the next iteration
        plt.clf()

# get_all_graph("mic")

# get_pistonball_MIC(pb_untrained, actor, ep_no)
# get_pistonball_MIC(pb_trained_10001, actor, ep_no)
# get_pistonball_MIC(pb_trained_10001, actor, 1)
# get_pistonball_MIC(pb_trained_10001, actor, 2)
# get_pistonball_MIC(pb_trained_10001, actor, 3)
# get_pistonball_MIC(pb_trained_10001, actor, 4)

# for actor in range(20):
#     if 10 <= actor < 15:
#     # get_pistonball_MIC_avg(pb_untrained25, actor, "mic")
#         get_pistonball_MIC_avg(pb_trained25, actor, "mic")
    
# get_pistonball_MIC_avg(pb_untrained25, actor, "pearson")
# get_pistonball_MIC_avg(pb_trained25, actor, "pearson")
# get_pistonball_MIC_avg(pb_untrained25, actor, "kendall")
# get_pistonball_MIC_avg(pb_trained25, actor, "kendall")
# get_pistonball_MIC_avg(pb_untrained25, actor, "spearman")
# get_pistonball_MIC_avg(pb_trained25, actor, "spearman")
# get_pistonball_MIC_avg(pb_untrained25, actor, "pearson")
# get_pistonball_MIC_avg(pb_trained25, actor, "pearson")
# get_pistonball_MIC_avg(pb_untrained25, actor, "pearson")
# get_pistonball_MIC_avg(pb_untrained25, actor, "mic")
# get_pistonball_MIC_avg(pb_trained25, actor, "pearson")
# get_pistonball_MIC_avg(pb_trained25, actor+1, "pearson")
# get_pistonball_MIC_avg(pb_trained25, actor+2, "pearson")
# get_pistonball_MIC_avg(pb_untrained25, 15, "pearson")
# get_pistonball_MIC_avg(pb_trained25, 15, "pearson")
# get_pistonball_MIC_avg(pb_trained25, actor, "mic")
# get_pistonball_MIC_avg(pb_untrained25, 10, "mic")
# get_pistonball_MIC_avg(pb_trained25, actor+1, "mic")
# get_pistonball_MIC_avg(pb_trained25, actor+2, "mic")
# get_pistonball_MIC_avg(pb_untrained25, 15, "mic")
# get_pistonball_MIC_avg(pb_trained25, 15, "mic")

def get_pistonball_MIC_avg(file_path, actor, stat="mic"):

    with open(file_path, 'rb') as file:
        trajectory_info = pickle.load(file)

    legend_label = os.path.splitext(os.path.basename(file_path))[0].split('last_')[-1]

    action_hist = {}
    num_episodes = len(trajectory_info)
    for ep in range(num_episodes):
        episode = np.array(trajectory_info[ep].cpu())
        test = np.random.randint(0,3,len(episode[actor]))
        print(episode[actor]==episode[actor])
        print(episode[actor])
        print(test)
        print(episode[actor].shape)
        mine = MINE(alpha=.9, c=15, est="mic_approx")
        mine.compute_score(episode[actor], episode[actor])
        tmine = MINE(alpha=.9, c=15, est="mic_approx")
        tmine.compute_score(test, test)
        print(mine.tic(norm=True))
        print(tmine.tic(norm=True))
    
# for actor in range(20):
#     get_pistonball_MIC_avg(pb_untrained100, actor, "mic")
#     get_pistonball_MIC_avg(pb_trained100, actor, "mic")
#     plt.show()

# def get_pistonball_MIC_alt(file_path, actor, ep_no):

#     with open(file_path, 'rb') as file:
#         trajectory_info = pickle.load(file)

#     # Extract the text between the last '_' and '.pk1' from the file path
#     legend_label = os.path.splitext(os.path.basename(file_path))[0].split('last_')[-1]

#     episode = trajectory_info[ep_no]
#     mine = MINE(alpha=.6, c=15, est="mic_approx")
#     mine.compute_score(episode[actor].cpu(), episode[actor].cpu())
#     print(f"{mine.mic()}")
#     action_hist = {}
#     for partner in range(episode.shape[0]):
#         mine.compute_score(episode[actor].cpu(), episode[partner].cpu())
#         print(f"({partner}: {mine.mic()}")
#         action_hist[partner] = mine.mic()
#     x = list(action_hist.keys())
#     y = list(action_hist.values())

#     # Plot lines from the actor's x,y data to all other xy data that is not the actor's
#     for i in range(len(x)):
#         if i != actor:
#             plt.plot([x[actor], x[i]], [y[actor], y[i]], linestyle='-', linewidth=1, color='black', alpha=0.1, zorder=0)

#     plt.xlabel('Agent ID #')
#     plt.ylabel('MIC Value')
#     plt.title(f'MIC Values for Agent {actor} and Partners')

#     # Create the scatter plot
#     plt.scatter(x, y, s=10, label=legend_label)
#     plt.xlim(19, -1)  # Keep the x-axis limits as before
#     plt.xticks(range(20), range(1, 21))  # Increase the value of tick labels by a factor of 1

#     # Add labels and title
#     plt.xlabel('Agent ID #')
#     plt.ylabel('MIC')
#     plt.title('Scatter Plot')

# # Show the plot
# actor = 10
# ep_no = 3
# get_pistonball_MIC_alt(pb_untrained, actor, ep_no)
# get_pistonball_MIC_alt(pb_trained_10001, actor, ep_no)
# get_pistonball_MIC(pb_trained_10001, actor, ep_no)
# plt.legend()

# plt.show()