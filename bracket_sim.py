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

st.title("March Madness Bracket Pool Simulator")
# st.header("What the hell is this?")

# st.markdown("This application allows you to upload your own bracket and simulate it against a pool of brackets generated using ESPN's public bracket picks data. The win probabilities for each matchup are generated via the GBM model that won the 2018 Women's NCAA Tournament and the 2022 Men's NCAA Tournament and has posted the lowest log-loss for any model ever posted on Kaggle's March Madness competitions (https://www.kaggle.com/code/jtrotman/winning-submission-improved-to-0-498).")

# st.header("How do I use this?")
# st.markdown("Enter the number of entries in your chosen pool, choose the pool entry fee (if it's a free pool use a tiny amount like 0.1), then download the sample CSVs and customize them according to the rules of your pool. Then upload the edited files and click Submit!")

# st.header("Can I just have the algorithm choose my bracket?")
# st.markdown("You can! Just choose 'I'm too lazy to submit my own picks' in the 'Import Custom Selections' dropdown and then you can simply choose the bracket with the best ROI from the table after the simulation is complete. If you lose, just blame Skynet")

# st.header("Who built this?")
# st.markdown("I did! If you have any questions you can contact me via twitter: https://twitter.com/blainejungwirth")
    
form = st.form("Sim Options")

with open('data/contest_info/payout_structure.csv') as f:
    st.download_button('Download Sample Contest CSV', f, 'payout_structure.csv')

with open('data/contest_info/sample_bracket.csv') as f:
    st.download_button('Download Sample Bracket Selections CSV', f, 'sample_bracket.csv')
    
with open('data/contest_info/points_structure.csv') as f:
    st.download_button('Download Sample Points CSV', f, 'points_structure.csv')

entries = form.number_input('Number of Pool Contestants',value=16)
entry_fee = form.number_input('Pool Entry Fee',value=20)
# Now add a submit button to the form:

@st.cache_data
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv(index=False).encode('utf-8')


uploaded_contest_file = form.file_uploader("Upload Contest CSV")
uploaded_points_file = form.file_uploader("Upload Pool Points Rules")

option = form.selectbox(
    'Import Custom Selections?',
     ["I have my own picks to upload","Too lazy to make my own picks"])

if option == 'I have my own picks to upload':
    uploaded_selection_file = form.file_uploader("Upload Selections CSV")
else:
    uploaded_selection_file = None

add_seeds_flag = form.radio("Does your Pool add points for the seeds of every winner?", ["Yes", "No"])
    
submitted = form.form_submit_button("Submit")   

df = pd.read_csv('data/public_picks.csv')
df = df.dropna()

spellings = pd.read_csv('data/kaggle/MTeamSpellings.csv', encoding='latin-1')

spellings['TeamNameSpelling'] = spellings['TeamNameSpelling'].str.replace('[^\w\s]',' ')
spellings = spellings.drop_duplicates(subset='TeamNameSpelling')
spellings = spellings.set_index('TeamNameSpelling').to_dict(orient='index')

whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

df.head()

df['Team'] = df['Team'].str.lower()

#fill the team dictionary with the team names and ids including the changed spellings above
team_name_fixes = {
    'bsu' : 'boise state',
    'colo' : 'colorado',
    'uva' : 'virginia',
    'csu': 'colorado state',
    'how': 'howard',
    'wag': 'wagner',
    'mtst': 'montana state',
    'gram': 'grambling',
    'miami': 'miami fl',
    'fau': 'florida atlantic',
    'msst': 'mississippi st',
    'amcc': 'a m corpus chris',
    'smo': 'se missouri st',
    'txso': 'tx southern',
    'fdu': 'fairleigh dickinson',
    'asu': 'arizona st',
    'nev': 'nevada',
    'texas amcc': 'texas am corpus christi',
    'tx christian': 'tcu',
    'n carolina': 'north carolina',
    'fla atlantic': 'florida atlantic',
    'wash state': 'washington st',
    'st marys': "saint-marys",
    's carolina': 'south carolina',
    'miss state': 'mississippi st',
    'james mad' : 'james madison',
    'grd canyon': 'grand canyon',
    'col charlestn': 'college of charleston',
    'lg beach st': 'long beach st',
    'st peters': "saint-peters",
    'texas a&m': 'texas am',
}

team_dict = {}
# Try to find the team in the spellings dictionary to get team IDs, print the teams that can't be found.
for i, r in df.iterrows():
    name = r['Team']
    seed = r['Seed']
    region = r['Region']
    first_four = '/' in name  # Check if the team is in the first four
    r64 = float(r['R64'].strip('%')) / 100
    r32 = float(r['R32'].strip('%')) / 100
    s16 = float(r['S16'].strip('%')) / 100
    e8 = float(r['E8'].strip('%')) / 100
    f4 = float(r['F4'].strip('%')) / 100
    ncg = float(r['NCG'].strip('%')) / 100
    # Handle the case for first four teams.
    if first_four:
        t1, t2 = name.split('/')
        t1 = t1.strip()
        t2 = t2.strip()
        t1 = team_name_fixes.get(t1, t1)  # Apply fixes if available and get the ID
        t2 = team_name_fixes.get(t2, t2)  # Apply fixes if available and get the ID
        t1_id = spellings.get(t1, {}).get('TeamID', None)  # Check if the team name is in the spellings dictionary and get the ID
        t2_id = spellings.get(t2, {}).get('TeamID', None)  # Check if the team name is in the spellings dictionary and get the ID
        if t1_id is None:
            print(f"Team {t1} not found in spellings.")
        if t2_id is None:
            print(f"Team {t2} not found in spellings.")
        # Add first team to the dictionary if it doesn't exist.
        print(t1,t1_id, t2, t2_id)
        if t1_id not in team_dict:
            team_dict[t1_id] = {
                'seed': seed,
                'name': t1,
                'id': t1_id,
                'region': region,
                'first_four': True,
                'own': {'R64': r64, 'R32': r32, 'S16': s16, 'E8': e8, 'F4': f4, 'NCG': ncg},
            }

        # Add second team to the dictionary if it doesn't exist.
        if t2_id not in team_dict:
            team_dict[t2_id] = {
                'seed': seed,
                'name': t2,
                'id': t2_id,
                'region': region,
                'first_four': True,
                'own': {'R64': r64, 'R32': r32, 'S16': s16, 'E8': e8, 'F4': f4, 'NCG': ncg},
            }
    else:
        # Apply fixes if available and get the ID.
        team_id = team_name_fixes.get(name, name)

        # Check if the team name is in the spellings dictionary and get the ID.
        if team_id in spellings:
            team_id = spellings[team_id]['TeamID']
        else:
            print(f"Team {name} not found in spellings.")
            continue  # Skip this team if not found.

        # Add the team to the dictionary if it doesn't exist.
        if team_id not in team_dict:
            team_dict[team_id] = {
                'seed_num': seed,
                'name': name,
                'id': team_id,
                'region': region,
                'first_four': False,
                'own': {'R64': r64, 'R32': r32, 'S16': s16, 'E8': e8, 'F4': f4, 'NCG': ncg},
            }
      
seeds = pd.read_csv('data/kaggle/MNCAATourneySeeds.csv')

seeds = seeds[seeds['Season']==2024]

print(team_dict.keys())

for i,r in seeds.iterrows():
    team_dict[r['TeamID']]['region'] = r['Seed'][0]
    team_dict[r['TeamID']]['seed'] = int(r['Seed'][1:3])
    if r['Seed'][-1] == 'a' or r['Seed'][-1] == 'b':
       team_dict[r['TeamID']]['first_four'] = True

    
class Team:
    def __init__(self, name, id, seed, region, first_four, own):
        if type(name) != str: raise TypeError("name needs to be of type str")
        if type(seed) != int: raise TypeError("seed needs to be of type int")
        self.name = name
        self.seed = seed
        self.region = region
        self.first_four = first_four
        self.own = {'R64': own['R64'], 'R32': own['R32'], 'S16': own['S16'], 'E8': own['E8'], 'F4': own['F4'], 'NCG': own['NCG'], 'first_four':0.5    }
        self.sim_results = {'first_four':0,'R64':0,'R32':0,'S16':0,'E8':0,'F4':0, 'NCG':0}
        self.eliminated = 0
        self.id = id
        self.seed_id = 0
        
    def __repr__(self):
        return f"<{self.seed} {self.name}>"    

teams = []
for k in team_dict.keys():
    t = Team(team_dict[k]['name'], team_dict[k]['id'], team_dict[k]['seed'], team_dict[k]['region'], team_dict[k]['first_four'], team_dict[k]['own'])
    teams.append(t)

matchup_probabilities = pd.read_csv('data/game_predictions.csv')
matchup_probabilities = matchup_probabilities.drop_duplicates()
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
                e.tourney_points[self.round_names[self.current_round]] += self.points_structure['Points'][self.round_names[self.current_round]]
                e.total_points[self.round_names[self.current_round]] += self.points_structure['Points'][self.round_names[self.current_round]]
                if self.add_seed_to_points:
                    e.tourney_points[self.round_names[self.current_round]] += winner.seed
                    e.total_points[self.round_names[self.current_round]] += winner.seed
                                      
        
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
                            self.games.append(Game(x, t, 2024, matchup_probabilities))
                            i+= 1
                        else:
                            self.games.append(Game(t, x, 2024, matchup_probabilities)) 
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
        self.user_submitted = False

if uploaded_points_file:
    points_df = pd.read_csv(uploaded_points_file)
    points_structure = points_df.set_index('Round').to_dict()
       
if submitted:
    # Can be used wherever a "file-like" object is accepted:
    payout_structure = pd.read_csv(uploaded_contest_file)
    contest_prize_structure ={}
    for i,r in payout_structure.iterrows():
        contest_prize_structure[i]= r['Payout']
    #st.write(payout_structure)

    tourney = Pool(2024, teams, entries, contest_prize_structure, points_structure, entry_fee)
    if add_seeds_flag == "Yes":
        tourney.add_seed_to_points = True
    tourney.generateBracketSelections()

    if uploaded_selection_file is not None:
        picks = pd.read_csv(uploaded_selection_file)    
        for c in picks.columns:
            tms = list(picks[c].dropna())
            ids = []
            for tm in tms:
                if tm not in spellings.keys():
                    print("Can't find " + tm + " please check file and resubmit")
                ids.append(spellings[tm]['TeamID'])
            tourney.entrants[0].selections[c] = ids
            tourney.entrants[0].user_submitted = True
            tourney.entrants[0].selection_names[c] = tms
    print('brackets created')

    sims = 10000
    tourney.monteCarlo(sims)
    print('tourney simmed')

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
    submissions = []

    for e in tourney.entrants:
        print(e.id, e.user_submitted)
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
        submissions.append(e.user_submitted)

    df = pd.DataFrame([ids, r64_selections,r32_selections,s16_selections,e8_selections,f4_selections, ncg_selections, r64_points, r32_points, s16_points, e8_points,f4_points,ncg_points,rewards, submissions])
    df = df.transpose()
    df.columns = ['id','r64_selections', 'r32_selections', 'r16_selections', 'e8_selections', 'f4_selections', 'ncg_selections', 'r64_points', 'r32_points', 's16_points', 'e8_points', 'f4_points', 'ncg_points', 'roi','user_submitted_bracket']

    st.dataframe(df)
    st.download_button(
    label="Download Pool Sims as CSV",
    data=convert_df_to_csv(df),
    file_name='pool_sims.csv',
    mime='text/csv',
    )
    
    ids = []
    names = []
    seeds = []
    regions = []
    r64_own = []
    r32_own = []
    s16_own = []
    e8_own = []
    f4_own = []
    ncg_own = []
    r64_sim_results = []
    r32_sim_results = []
    s16_sim_results = []
    e8_sim_results = []
    f4_sim_results = []
    ncg_sim_results = []
    for t in tourney.teams:
        ids.append(t.id)
        names.append(t.name)
        seeds.append(t.seed)
        regions.append(t.region)
        r64_own.append(t.own['R64'])
        r32_own.append(t.own['R32'])
        s16_own.append(t.own['S16'])
        e8_own.append(t.own['E8'])
        f4_own.append(t.own['F4'])
        ncg_own.append(t.own['NCG'])
        r64_sim_results.append(t.sim_results['R64']/sims)
        r32_sim_results.append(t.sim_results['R32']/sims)
        s16_sim_results.append(t.sim_results['S16']/sims)
        e8_sim_results.append(t.sim_results['E8']/sims)
        f4_sim_results.append(t.sim_results['F4']/sims)
        ncg_sim_results.append(t.sim_results['NCG']/sims)
    teams_df = pd.DataFrame([ids,     names ,    seeds ,    regions ,    r64_sim_results ,
    r32_sim_results ,
    s16_sim_results ,
    e8_sim_results ,
    f4_sim_results,
    ncg_sim_results,
    r64_own ,
    r32_own ,
    s16_own ,
    e8_own ,
    f4_own ,
    ncg_own ])      
    teams_df = teams_df.transpose()
    teams_df.columns = ['id','name','seed','region', 'R64 Sim Win Prob', 'R32 Sim Win Prob', 'S16 Sim Win Prob' , 'E8 Sim Win Prob', 'F4 Sim Win Prob', 'Champ Sim Win Prob', 'R64 Public Pick %', 'R32 Public Pick %', 'S16 Public Pick %','E8 Public Pick %','F4 Public Pick %','Champ Public Pick %']
    
    teams_df['R64 Leverage'] = teams_df['R64 Sim Win Prob'] - teams_df['R64 Public Pick %']
    teams_df['R32 Leverage'] = teams_df['R32 Sim Win Prob'] - teams_df['R32 Public Pick %']
    teams_df['S16 Leverage'] = teams_df['S16 Sim Win Prob'] - teams_df['S16 Public Pick %']
    teams_df['E8 Leverage'] = teams_df['E8 Sim Win Prob'] - teams_df['E8 Public Pick %']
    teams_df['F4 Leverage'] = teams_df['F4 Sim Win Prob'] - teams_df['F4 Public Pick %']
    teams_df['Champ Leverage'] = teams_df['Champ Sim Win Prob'] - teams_df['Champ Public Pick %']
    
    st.dataframe(teams_df)
    
    st.download_button(
    label="Download Team Results as CSV",
    data=convert_df_to_csv(teams_df),
    file_name='team_results.csv',
    mime='text/csv',
    )
    
