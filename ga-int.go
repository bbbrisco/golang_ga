package main

import "bufio"
import "fmt"
import "os"
import "strings"

import "github.com/bbbrisco/ga"

func count_bits_set(b byte) uint {
    set := uint(0)
    for b != 0 {
	if (b & 1) != 0 {
	    set++
	}
	b >>= 1
    }
    return set
}

func bitcnt_fitness(gene_data []byte) float64 {
    bit_count := uint(0)
    for _, curr := range gene_data {
	bit_count += count_bits_set(curr)
    }
    return float64(bit_count)
}

func main() {
    var ga_int ga.GA
    var params ga.GA_Params

    reader := bufio.NewReader(os.Stdin)

    params.Keep_top_n = 5
    params.Crossover_rate = 0.65
    params.Mutation_rate = 0.002

    bit_size := uint(1024)

    ga_int.Init(bit_size, 64, bitcnt_fitness, params)
    done := false
    for !done {
	best_fitness := ga_int.Population[0].Fitness
	if (best_fitness >= float64(bit_size)) {
	    fmt.Printf("optimal solution reached on generation %d\n", ga_int.Generation)
	    done = true
	    continue
	}
	fmt.Printf("ga-int[%d,%f]$ ", ga_int.Generation, best_fitness)
	cmd, _ := reader.ReadString('\n')
	// fmt.Printf("%s\n", cmd)
	// fmt.Printf("got command '%s'\n", cmd)
	if strings.HasPrefix(cmd, "quit") {
	    done = true
	    continue
	}
	if strings.HasPrefix(cmd, "print") {
	    var arg int
	    i, err := fmt.Sscanf(cmd, "print %d\n", &arg)
	    if (i != 1) || (err != nil) {
		fmt.Printf("error: usage 'print <d>'\n")
		fmt.Printf("error:   matched = %d, err = '%s'\n", i, err.Error())
		continue
	    }
	    pop_len := int(len(ga_int.Population))
	    if (arg < 0) || (arg > pop_len) {
		fmt.Printf("only %d genes in population - truncating list...\n", pop_len)
		arg = pop_len
	    }
	    fmt.Printf("dumping %d genes...\n", arg)
	    for j := 0; j < arg; j++ {
		ind := ga_int.Population[j]
		fmt.Printf("%4d: f=%2.1f [%s]\n", j, ind.Fitness, ind.BitString())
	    }
	    continue
	}

	if strings.HasPrefix(cmd, "run") {
	    var arg int
	    i, err := fmt.Sscanf(cmd, "run %d\n", &arg)
	    if (i != 1) || (err != nil) {
		fmt.Printf("error: usage 'run <d>'\n")
		fmt.Printf("error:   matched = %d, err = '%s'\n", i, err.Error())
		continue
	    }
	    for i = 0; i < arg; i++ {
		ga_int.Evolve();
	    }
	    continue;
	}

	if strings.HasPrefix(cmd, "stats") {
	    fmt.Printf("%20s %20s %20s\n", "Generation", "Best Fitness", "Avg Fitness")
	    for i := uint(0); i < ga_int.Generation; i++ {
		fmt.Printf("%20d %20.2f %20.2f\n", i, ga_int.Stats_best[i], ga_int.Stats_avg[i])

	    }
	    continue
	}

	if strings.HasPrefix(cmd, "help") {
	    fmt.Printf("commands:\n")
	    fmt.Printf("\t%20s -- print this list\n", "help")
	    fmt.Printf("\t%20s -- print (up to) top <d> genes\n", "print <d>")
	    fmt.Printf("\t%20s -- run <d> generations\n", "run <d>")
	    fmt.Printf("\t%20s -- dump some cumulative stats\n", "stats")
	    fmt.Printf("\t%20s -- terminate program\n", "quit")
	}
    }
}
