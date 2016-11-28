package ga

// import "fmt"

import "fmt"
import "math/rand"
import "os"
import "sort"
import "time"

type FitnessVal float64
type FitnessFunc func([]byte) float64
type GenePool	[]*Individual

type Individual struct {
    Data	[]byte
    Selected	bool
    Fitness	float64
    Cum_prob	float64
}

type GA_Params struct {
    Crossover_rate float64
    Mutation_rate float64
    Keep_top_n	uint
}

type GA struct {
    Population	GenePool
    Params	GA_Params
    Generation	uint
    data_size	uint
    fit_func	FitnessFunc
    Stats_best	[]float64
    Stats_avg	[]float64
}

func (ind *Individual) BitString() string {
    string_rep := make([]rune, len(ind.Data)*8)
    for i, v := range(ind.Data) {
	mask := byte(0x80)
	for j := 0; j < 8; j++ {
	    bit_rune := '0'
	    if (v & mask) != 0 {
		bit_rune = '1'
	    }
	    string_rep[i*8+j] = bit_rune
	    mask >>= 1
	}
    }
    return string(string_rep)
}

// some functions to enable us to sort our population of individuals
func (pool GenePool) Len() int {
    return len(pool)
}

func (pool GenePool) Less(i, j int) bool {
    return pool[i].Fitness > pool[j].Fitness
}

func (pool GenePool) Swap(i, j int) {
    pool[i], pool[j] = pool[j], pool[i]
}

func (ga *GA) MeasureAndSort() {
    for _, ind := range ga.Population {
	ind.Fitness = ga.fit_func(ind.Data)
    }
    sort.Sort(ga.Population)
}

// pool_size : number of individuals in our gene_pool/population
// data_size : number of bits of (initially random) data in each individual
func (ga *GA) Init(pool_size uint, data_size uint, fit_func FitnessFunc, p GA_Params) {

    rand.Seed(time.Now().UTC().UnixNano())
    data_bytes := (data_size + 7) / 8
    ga.Population = make(GenePool, pool_size)
    // for _, ind := range ga.Population {
    for i := range ga.Population {
	// var ind *Individual
	ind := new(Individual)
	ind.Data = make([]byte, data_bytes)
	random_word := rand.Uint32()
	for j := range ind.Data {
	    if (j % 4) == 0 {
		random_word = rand.Uint32()
	    }
	    ind.Data[j] = byte(random_word & 0xff)
	    random_word >>= 8
	}
	ga.Population[i] = ind
    }
    ga.Params = p
    ga.data_size = data_size
    ga.Generation = 0
    ga.fit_func = fit_func
    ga.MeasureAndSort()
    ga.Stats_best = make([]float64, 0, 1024)
    ga.Stats_avg = make([]float64, 0, 1024)
    ga.Stats_best = append(ga.Stats_best, ga.Population[0].Fitness)
    ga.Stats_avg = append(ga.Stats_avg, ga.AvgFitness())
}

func (ga *GA) AvgFitness() float64 {
    total := 0.0
    for _, v := range ga.Population {
	total += float64(v.Fitness)
    }
    return total/float64(len(ga.Population))
}

func (ga *GA) Crossover(p1 *Individual, p2 *Individual) (c1 *Individual, c2 *Individual) {
    // use single point crossover
    data_bytes := uint(len(p1.Data))
    c1 = new(Individual)
    c2 = new(Individual)
    c1.Data = make([]byte, data_bytes)
    c2.Data = make([]byte, data_bytes)
    bit_count := uint(data_bytes*8)
    crossover_bit  := uint(rand.Uint32()) % bit_count
    crossover_byte := crossover_bit / 8
    for i := uint(0); i < data_bytes; i++ {
	if i < crossover_byte {
	    c1.Data[i] = p1.Data[i]
	    c2.Data[i] = p2.Data[i]
	} else {
	    if i > crossover_byte {
		c1.Data[i] = p2.Data[i]
		c2.Data[i] = p1.Data[i]
	    } else {
		var b1 byte = 0
		var b2 byte = 0
		k := (crossover_bit - (8*crossover_byte))
		mask := byte((1 << (8-k)) - 1)
		b1 = p1.Data[i] & mask
		b2 = p2.Data[i] & mask
		b1 |= p2.Data[i] & ^mask
		b2 |= p1.Data[i] & ^mask
		c1.Data[i] = b1
		c2.Data[i] = b2
	    }
	}
    }
    // fmt.Fprintf(os.Stdout, "Crossover: p1 = %s\n", p1.BitString())
    // fmt.Fprintf(os.Stdout, "Crossover: p2 = %s\n", p2.BitString())
    // fmt.Fprintf(os.Stdout, "Crossover: c1 = %s\n", c1.BitString())
    // fmt.Fprintf(os.Stdout, "Crossover: c2 = %s\n", c2.BitString())
    // fmt.Fprintf(os.Stdout, "Crossover: bit = %d, byte = %d\n", crossover_bit, crossover_byte)
    return c1, c2
}

func (ga *GA) Evolve() {
    var ind *Individual
    var i uint

    // assume current population is measured for fitness and sorted
    // create a new pool to represent next generation
    next_gen := make([]*Individual, 0, len(ga.Population))

    // clear selected bits (used to keep an individual from being copied > 1 time to new generation)
    for _, ind = range(ga.Population) {
	ind.Selected = false
    }

    // keep top n
    if ga.Params.Keep_top_n > 0 {
	for i = uint(0); i < ga.Params.Keep_top_n; i++ {
	    next_gen = append(next_gen, ga.Population[i])
	    ga.Population[i].Selected = true
	}
    }
    total_fitness := 0.0
    for _, ind = range ga.Population {
	total_fitness += ind.Fitness
    }

    // for each ind, cum probability -> prev->cum_prob + Fitness/total_fitness
    cum_prob := 0.0
    for _, ind = range ga.Population {
	cum_prob += ((ind.Fitness)/total_fitness)
	ind.Cum_prob = cum_prob
    }

    pop_target := len(ga.Population)
    for len(next_gen) < pop_target {

	// spin the roulette wheel for 1st parent (p1)
	r1 := rand.Float64()
	var p1 *Individual = nil
	for _, ind = range ga.Population {
	    if ind.Cum_prob >= r1 {
		p1 = ind
		break
	    }
	}
	if p1 == nil {
	    panic("parent1 not found\n")
	}

	// spin the roulette wheel for 2nd parent (p2)
	r2 := rand.Float64()
	var p2 *Individual = nil
	for _, ind = range ga.Population {
	    if ind.Cum_prob >= r2 {
		// should we allow p1 == p2?
		if p2 == p1 {
		     continue
		}
		p2 = ind
		break
	    }
	}
	if p2 == nil {
	    panic("parent2 not found\n")
	}
	crand := rand.Float64()
	// fmt.Fprintf(os.Stdout, "gene selection: len(next_gen) = %d, crand=%f\n", len(next_gen), crand)
	if crand <= ga.Params.Crossover_rate {
	    c1, c2 := ga.Crossover(p1, p2)
	    next_gen = append(next_gen, c1)
	    if len(next_gen) < pop_target {
		next_gen = append(next_gen, c2)
	    }
	} else {
	    // we could end up adding individuals multiple times to next_gen
	    // should we set a selected bit and skip if already set? 
	    // yes... apparently we should
	    // fmt.Fprintf(os.Stdout, "p1.Selected = %t, p2.Selected = %t\n", p1.Selected, p2.Selected)
	    // fmt.Fprintf(os.Stdout, "copying parents %s and %s\n", p1.BitString(), p2.BitString())
	    if p1.Selected != true {
		next_gen = append(next_gen, p1)
		p1.Selected = true
	    }
	    if len(next_gen) < pop_target {
		if p2.Selected != true {
		    next_gen = append(next_gen, p2)
		    p2.Selected = true
		}
	    }
	}
    }

    var p int
    for p, ind = range next_gen {
	for j := range ind.Data {
	    mask := byte(0x80)
	    for k := uint(0); k < 8; k++ {
		if rand.Float64() < ga.Params.Mutation_rate {
		    // fmt.Fprintf(os.Stdout, "mutating bit %d in gene %d\n", uint(j)*8+k, p)
		    ind.Data[j] ^= mask
		}
		mask >>= 1
	    }
	}
    }
    ga.Population = next_gen
    ga.MeasureAndSort()
    ga.Generation++
    ga.Stats_best = append(ga.Stats_best, ga.Population[0].Fitness)
    ga.Stats_avg = append(ga.Stats_avg, ga.AvgFitness())
}
