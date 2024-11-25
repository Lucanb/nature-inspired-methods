from genetic_class import GA

moea = GA(file_name='a280-n279')
moea.population_generator(length=100)
moea.optimize(generations=5, tournament_size=5, crossover = 'OX', selection = 'tournament', mutation = 'inversion', replacement = 'non-elitist')
moea.export_result()