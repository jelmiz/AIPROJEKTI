import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make(game='SuperMarioBros3-Nes')

imgarray = []

def eval_genomes(genomes,config):

    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        ypos = 0
        jump = 0
        xpos_max = 0
        jumpBool = True
        done = False

        while not done:

            env.render()
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            for x in ob:
                for y in x:
                    imgarray.append(y)
            
            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)

            imgarray.clear()
            
            #DATA.json TIEDOSTOSTA ARVOT
            xpos = info['mario_x_pos']
            jump = info['inair']
            lives = info['lives']
            ypos = info['mario_y_pos']
            
            #PALKINTO MÄÄRITTELYT
            #Jos hyppää saa reward x 2, hypyn arvoksi tulee false, jottei saa pisteitä ilmassa olemisesta
            if jump == 1 and jumpBool == True:
                fitness_current += rew * 2
                jumpBool = False
            #Jos ei maassa saa reward (maassa arvo 128)
            if ypos < 128:
                fitness_current += rew
            #Jos yli 130 koordinaatin x arvona saa reward x 2
            if xpos > 130:
                fitness_current += rew * 2
            #X koordinaatin arvo on alussa 24 ja alkaa 0:sta kun ylittää arvon 255 muistin takia
            if xpos == 0:
                fitness_current += 255
                xpos_max = 0
            #Jos xpos arvo kasvaa niin saa rewardin. Lisäksi liike resetoi hypystä saatavat rewardit
            if xpos > xpos_max and jump == 0:
                fitness_current += rew
                xpos_max = xpos
                jumpBool = True
            
            #Jos fitness kasvaa, peli jatkuu. Muussa tapauksessa aloittaa ajastimen.
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            #Jos ajastin pääsee 350 tai Mario menettää elämän, tekoäly aloittaa uuden yrityksen
            if done or counter == 350 or lives == 3:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, 
neat.DefaultStagnation, 'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

#with open('winner.pkl', 'wb') as output:
    #pickle.dump(winner,output, 1)

