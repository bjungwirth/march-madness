from xml.etree.ElementTree import QName
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re 
import itertools
import random
import operator
import numpy as np
import streamlit as st

with open('data/contest_info/payout_structure.csv') as f:
    st.download_button('Download Sample Contest CSV', f, 'payout_structure.csv')

with open('data/contest_info/sample_bracket.csv') as f:
    st.download_button('Download Sample Bracket Selections CSV', f, 'sample_bracket.csv')

entries = st.sidebar.number_input('Number of Pool Contestants',value=16)
entry_fee = st.sidebar.number_input('Pool Entry Fee',value=20)

uploaded_contest_file = st.file_uploader("Upload Contest CSV")

option = st.selectbox(
    'Import Custom Selections?',
     ["I have my own picks to upload","Too lazy to make my own picks"])

if option == 'I have my own picks to upload':
    uploaded_selection_file = st.file_uploader("Upload Selections CSV")
else:
    uploaded_selection_file = None
url = 'https://fantasy.espn.com/tournament-challenge-bracket/2023/en/whopickedwhom'
# Create object page
page = requests.get(url)

soup = BeautifulSoup(page.text, 'html.parser')

# Creating list with all tables
tables = soup.find_all('table')

#  Looking for the table with the classes 'wikitable' and 'sortable'
table = soup.find('table', class_='wpw-table')

# Defining of the dataframe
df = pd.read_html(str(table))[0]

spellings = pd.read_csv('data/kaggle/MTeamSpellings.csv', encoding='latin-1')

spellings['TeamNameSpelling'] = spellings['TeamNameSpelling'].str.replace('[^\w\s]',' ')
spellings = spellings.drop_duplicates(subset='TeamNameSpelling')
spellings = spellings.set_index('TeamNameSpelling').to_dict(orient='index')

whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
team_dict = {}
for i,r in df.iterrows():
    for col in r.index:
        seed = int(re.findall(r'^(\d{1,2})', r[col])[0])
        pct = float(re.findall(r'-(\d{1,2}.\d)', r[col])[0])
        if re.search(r'/',r[col]):
            names = re.split(r'/',r[col])
            pct = pct/2
            for name in names:
                name = ''.join(filter(whitelist.__contains__, name)).lower()
                if name == 'miami':
                    name = 'miami fl'
                if name == 'fau':
                    name = 'florida atlantic'
                if name == 'msst':
                    name = 'mississippi st'
                if name == 'amcc':
                    name = 'a m corpus chris'
                if name == 'smo':
                    name = 'se missouri st'
                if name == 'txso':
                    name = 'tx southern'
                if name == 'fdu':
                    name = 'fairleigh dickinson'
                if name == 'asu':
                    name = 'arizona st'
                if name == 'nev':
                    name = 'nevada'
                id = spellings[name]['TeamID']
                if id not in team_dict.keys():
                    team_dict[id] = {'seed':seed,'name':name, 'id':id, 'region':'','first_four':True}
                team_dict[id][col] = {'espn_own':pct}     
        else:
            name = ''.join(filter(whitelist.__contains__, r[col])).lower()
            if name == 'miami':
                name = 'miami fl'
            if name == 'fau':
                name = 'florida atlantic'
            if name == 'msst':
                name = 'mississippi st'
            if name == 'amcc':
                name = 'a m corpus chris'
            if name == 'smo':
                name = 'se missouri st'
            if name == 'txso':
                name = 'tx southern'
            if name == 'fdu':
                name = 'fairleigh dickinson'
            if name == 'asu':
                name = 'arizona st'
            if name == 'nev':
                name = 'nevada'
            id = spellings[name]['TeamID']
            if id not in team_dict.keys():
                team_dict[id] = {'seed':seed,'name':name, 'id':id, 'region':'','first_four':False}
            team_dict[id][col] = {'espn_own':pct}

seeds = pd.read_csv('data/kaggle/MNCAATourneySeeds.csv')

seeds = seeds[seeds['Season']==2023]

for i,r in seeds.iterrows():
    team_dict[r['TeamID']]['region'] = r['Seed'][0]
    team_dict[r['TeamID']]['seed'] = int(r['Seed'][1:3])
    if r['Seed'][-1] == 'a' or r['Seed'][-1] == 'b':
       team_dict[r['TeamID']]['first_four'] = True

    
class Team:
    def __init__(self, name, id, seed, region, first_four):
        if type(name) != str: raise TypeError("name needs to be of type str")
        if type(seed) != int: raise TypeError("seed needs to be of type int")
        self.name = name
        self.seed = seed
        self.region = region
        self.first_four = first_four
        self.own = {'R64':0,'R32':0,'S16':0,'E8':0,'F4':0, 'NCG':0}
        self.sim_results = {'first_four':0,'R64':0,'R32':0,'S16':0,'E8':0,'F4':0, 'NCG':0}
        self.eliminated = 0
        self.id = id
        self.seed_id = 0
        
    def __repr__(self):
        return f"<{self.seed} {self.name}>"    

teams = []
for k in team_dict.keys():
    t = Team(team_dict[k]['name'], team_dict[k]['id'], team_dict[k]['seed'], team_dict[k]['region'], team_dict[k]['first_four'])
    t.own['R64'] = team_dict[k]['R64']['espn_own']
    t.own['R32'] = team_dict[k]['R32']['espn_own']
    t.own['S16'] = team_dict[k]['S16']['espn_own']
    t.own['E8'] = team_dict[k]['E8']['espn_own']
    t.own['F4'] = team_dict[k]['F4']['espn_own']
    t.own['NCG'] = team_dict[k]['NCG']['espn_own']
    teams.append(t)

matchup_probabilities = pd.read_csv('data/game_predictions.csv')
matchup_probabilities = matchup_probabilities.set_index('ID').to_dict(orient='index')

round_names = {
    0: 'First Four',
    1: 'First Round',
    2: 'Round of 32',
    3: 'Sweet 16',
    4: 'Elite 8',
    5: 'Final 4',
    6: 'Championship'
}

points_structure = {
    0: 0,
    1: 10,
    2: 20,
    3: 40,
    4: 80,
    5: 160,
    6: 320
}


class Pool():
    def __init__(self, year, teams, contest_entrants, contest_prize_structure, points_structure, entry_fee):
        self.current_round = 0
        self.year = year
        self.teams = teams
        self.contest_entrants = int(contest_entrants)
        self.entry_fee = entry_fee
        self.contest_prize_structure = contest_prize_structure
        self.load_contest_structure()
        self.add_seed_to_points = False
        self.round_names = {
            0: 'first_four',
            1: 'R64',
            2: 'R32',
            3: 'S16',
            4: 'E8',
            5: 'F4',
            6: 'NCG'
        }
        self.points_structure = points_structure
        #self.region_dict = {'W':'WX', 'X':'WX', 'Y':'YZ', 'Z':'YZ'}
        self.regions = ['W','X','Y','Z']
        self.seed_order = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
        self.bracket = []
        self.entrants = []
        self.simFirstFour()
        self.populateBracket()
    
    def resetEntrants(self):
        for e in self.entrants:
            e.tourney_points = {'R64':0, 'R32':0,'S16':0,'E8':0,'F4':0, 'NCG':0}
            
    def load_contest_structure(self):
        l_array = np.full(shape=self.contest_entrants - len(self.contest_prize_structure), fill_value=-self.entry_fee)
        payout_array = np.array(list(self.contest_prize_structure.values()))
        self.payout_array = np.concatenate((payout_array, l_array))
        
    def simFirstFour(self):
        ff = FirstFour(self.teams)
        for t in ff.games:
            t.simMatchup('first_four')   
        self.current_round += 1
    
    def populateBracket(self):
        i=0
        for r in self.regions:
            for s in self.seed_order:
                for t in self.teams:
                    if t.region == r and t.seed == s and t.eliminated ==0:
                        t.seed_id = i
                        self.bracket.append(t)  
                        i+=1
                        
    def monteCarlo(self, sims):
        for i in range(sims):
            self.simTourney()
            self.resetTourney()
            self.resetEntrants()
            #print(i)
        
    def simTourney(self):
        while self.current_round <= 6:
            self.SimulateRound()
            self.current_round += 1
        self.calculatePrizes()
            
    def resetTourney(self):
        for e in self.teams:
            e.eliminated = 0
        self.current_round = 0
        self.bracket = []
        self.simFirstFour()
        self.populateBracket()

    def setUpRoundMatchups(self):
        entrants_remaining = []
        for i,e in enumerate(self.bracket):
            if e.eliminated == 0:
                #print(e.eliminated, e.name, e.seed_id)
                entrants_remaining.append(e)
        entrants_remaining.sort(key=lambda x: x.seed_id, reverse=False)  
        return entrants_remaining
        
    def SimulateRound(self):
        entrants_remaining = self.setUpRoundMatchups()
        for v, w in zip(entrants_remaining[::2],entrants_remaining[1::2]):
            if w.id > v.id:
                self.simulateMatchup(v,w, self.year, matchup_probabilities)  
            else:
                self.simulateMatchup(w,v, self.year, matchup_probabilities)
                
    def simulateMatchup(self, t1,t2, year, matchup_probabilities):
        ids = [t1.id, t2.id]
        matchup_str = str(self.year) + '_' + str(ids[0]) + '_' + str(ids[1])
        probs = matchup_probabilities[matchup_str]
        r = random.random()
        if r <= probs['Pred']:
            winner = t1
            loser = t2
        else:
            winner = t2
            loser = t1
        winner.sim_results[self.round_names[self.current_round]]+= 1
        loser.eliminated = 1       
        for e in self.entrants:
            if winner.id in e.selections[self.round_names[self.current_round]]:
                e.tourney_points[self.round_names[self.current_round]] += self.points_structure[self.current_round]
                e.total_points[self.round_names[self.current_round]] += self.points_structure[self.current_round]
        
    def generateBracketSelections(self):
        for i in range(self.contest_entrants):
            entrant = Entrant(i)
            while self.current_round <= 6:
                entrants_remaining = []
                for i,e in enumerate(self.bracket):
                    if e.eliminated == 0:
                        #print(e.eliminated, e.name, e.seed_id)
                        entrants_remaining.append(e)
                entrants_remaining.sort(key=lambda x: x.seed_id, reverse=False)  
                for t1, t2 in zip(entrants_remaining[::2],entrants_remaining[1::2]):
                    own = np.array([t1.own[self.round_names[self.current_round]], t2.own[self.round_names[self.current_round]]])
                    own = own / np.sum(own)
                    r = random.random()
                    if r <= own[0]:
                        winner = t1
                        loser = t2
                    else:
                        winner = t2
                        loser = t1      
                    entrant.selections[self.round_names[self.current_round]].append(winner.id)
                    entrant.selection_names[self.round_names[self.current_round]].append(winner.name)
                    loser.eliminated = 1  
                self.current_round += 1        
            self.resetTourney()
            self.entrants.append(entrant)
            
    def calculatePrizes(self):
        standings_tuple = []
        for e in self.entrants:
            pts = sum(e.tourney_points.values())
            standings_tuple.append((pts, e))
        standings_tuple.sort(key=lambda a: a[0], reverse=True)
        i = 0
        for s in standings_tuple:
            s[1].rewards += self.payout_array[i]
            i+=1
            s[1].finish_positions.append(i+1)

class Game:
    def __init__(self, team1, team2, year, matchup_probabilities):
        self.team1 = team1
        self.team2 = team2
        self.year = year
        self.probs = self.getMatchupProbabiltiies(matchup_probabilities)
        self.order_flipped = False
    
    def getMatchupProbabiltiies(self, matchup_probabilities):
        ids = [self.team1.id, self.team2.id]          
        matchup_str = str(self.year) + '_' + str(ids[0]) + '_' + str(ids[1])
        return matchup_probabilities[matchup_str]
    
    def simMatchup(self,round):
        r = random.random()
        if r <= self.probs['Pred']:
            winner = self.team1
            loser = self.team2
        else:
            winner = self.team2
            loser = self.team1
        winner.sim_results[round]+= 1
        loser.eliminated = 1
            
class FirstFour:
    def __init__(self, teams):
        self.teams = []
        self.fillTeams(teams)
        self.games = []
        self.fillGames()
        self.id = 0
    
    def fillTeams(self, teams):
        for t in teams:
            if t.first_four == True:
                self.teams.append(t)
                
    def fillGames(self):
        i=0
        used_teams = []
        for t in self.teams:
            if t not in used_teams:
                for x in self.teams:
                    if x.region == t.region and x.seed == t.seed and x.id != t.id and x.id not in used_teams:
                        if t.id > x.id:
                            self.games.append(Game(x, t, 2023, matchup_probabilities))
                            i+= 1
                        else:
                            self.games.append(Game(t, x, 2023, matchup_probabilities)) 
                            i+= 1
                        used_teams.append(x)
            used_teams.append(t.id)     

class Entrant:
    def __init__(self, id):
        self.id = id
        self.selections = {'R64':[], 'R32':[],'S16':[],'E8':[],'F4':[], 'NCG':[]}
        self.selection_names = {'R64':[], 'R32':[],'S16':[],'E8':[],'F4':[], 'NCG':[]}
        self.tourney_points = {'R64':0, 'R32':0,'S16':0,'E8':0,'F4':0, 'NCG':0}
        self.rewards = 0
        self.total_points = {'R64':0, 'R32':0,'S16':0,'E8':0,'F4':0, 'NCG':0}
        self.finish_positions = []
                
    def uploadPicks(self):
        return True
    
if uploaded_selection_file is not None:
    print('yeehaw')     
           
if uploaded_contest_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    payout_structure = pd.read_csv(uploaded_contest_file)
    #st.write(payout_structure)

    contest_prize_structure ={}
    for i,r in payout_structure.iterrows():
        contest_prize_structure[i]= r['Payout']

    tourney = Pool(2023, teams, entries, contest_prize_structure, points_structure, entry_fee)

    tourney.generateBracketSelections()

    tourney.simTourney()

    sims = 10000
    tourney.monteCarlo(sims)

    r64_selections = []
    r32_selections = []
    s16_selections = []
    e8_selections = []
    f4_selections = []
    ncg_selections = []
    r64_points = []
    r32_points = []
    s16_points = []
    e8_points = []
    f4_points = []
    ncg_points = []
    rewards = []
    ids = []

    for e in tourney.entrants:
        r64_selections.append(e.selection_names['R64'])
        r32_selections.append(e.selection_names['R32'])
        s16_selections.append(e.selection_names['S16'])
        e8_selections.append(e.selection_names['E8'])
        f4_selections.append(e.selection_names['F4'])
        ncg_selections.append(e.selection_names['NCG'])
        r64_points.append(e.total_points['R64']/sims)
        r32_points.append(e.total_points['R32']/sims)
        s16_points.append(e.total_points['S16']/sims)
        e8_points.append(e.total_points['E8']/sims)
        f4_points.append(e.total_points['F4']/sims)
        ncg_points.append(e.total_points['NCG']/sims)
        rewards.append(e.rewards/sims)
        ids.append(e.id)

    df = pd.DataFrame([ids, r64_selections,r32_selections,s16_selections,e8_selections,f4_selections, ncg_selections, r64_points, r32_points, s16_points, e8_points,f4_points,ncg_points,rewards])
    df = df.transpose()
    df.columns = ['id','r64_selections', 'r32_selections', 'r16_selections', 'e8_selections', 'f4_selections', 'ncg_selections', 'r64_points', 'r32_points', 's16_points', 'e8_points', 'f4_points', 'ncg_points', 'roi']

    st.dataframe(df)