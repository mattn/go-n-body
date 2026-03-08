/* The Computer Language Benchmarks Game
 * https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
 *
 * contributed by The Go Authors.
 * based on C program by Christoph Bauer
 * Optimized with Go 1.26 SIMD (GOEXPERIMENT=simd)
 */

package main

import (
	"flag"
	"fmt"
	"math"
	"simd/archsimd"
	"strconv"
)

var n = 0

const (
	solarMass   = 4 * math.Pi * math.Pi
	daysPerYear = 365.24
	N           = 5
	PAIRS       = N * (N - 1) / 2
)

var (
	pos  [N]archsimd.Float64x4
	vel  [N]archsimd.Float64x4
	mass [N]float64
)

func initBodies() {
	zero := archsimd.BroadcastFloat64x4(0)
	mass[0] = solarMass
	pos[0] = zero
	vel[0] = zero

	mass[1] = 9.54791938424326609e-04 * solarMass
	pos[1] = archsimd.LoadFloat64x4(&[4]float64{0, 4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01})
	vel[1] = archsimd.LoadFloat64x4(&[4]float64{0, 1.66007664274403694e-03 * daysPerYear, 7.69901118419740425e-03 * daysPerYear, -6.90460016972063023e-05 * daysPerYear})

	mass[2] = 2.85885980666130812e-04 * solarMass
	pos[2] = archsimd.LoadFloat64x4(&[4]float64{0, 8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01})
	vel[2] = archsimd.LoadFloat64x4(&[4]float64{0, -2.76742510726862411e-03 * daysPerYear, 4.99852801234917238e-03 * daysPerYear, 2.30417297573763929e-05 * daysPerYear})

	mass[3] = 4.36624404335156298e-05 * solarMass
	pos[3] = archsimd.LoadFloat64x4(&[4]float64{0, 1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01})
	vel[3] = archsimd.LoadFloat64x4(&[4]float64{0, 2.96460137564761618e-03 * daysPerYear, 2.37847173959480950e-03 * daysPerYear, -2.96589568540237556e-05 * daysPerYear})

	mass[4] = 5.15138902046611451e-05 * solarMass
	pos[4] = archsimd.LoadFloat64x4(&[4]float64{0, 1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01})
	vel[4] = archsimd.LoadFloat64x4(&[4]float64{0, 2.68067772490389322e-03 * daysPerYear, 1.62824170038242295e-03 * daysPerYear, -9.51592254519715870e-05 * daysPerYear})
}

func offsetMomentum() {
	o := archsimd.BroadcastFloat64x4(0)
	for i := 0; i < N; i++ {
		o = o.Add(vel[i].Mul(archsimd.BroadcastFloat64x4(mass[i])))
	}
	vel[0] = o.Mul(archsimd.BroadcastFloat64x4(-1.0 / solarMass))
}

func energy() float64 {
	var e float64
	for i := 0; i < N; i++ {
		v := vel[i]
		vsq := v.Mul(v)
		var buf [4]float64
		vsq.Store(&buf)
		e += 0.5 * mass[i] * (buf[1] + buf[2] + buf[3])
		for j := i + 1; j < N; j++ {
			d := pos[i].Sub(pos[j])
			dsq := d.Mul(d)
			var dbuf [4]float64
			dsq.Store(&dbuf)
			dist := math.Sqrt(dbuf[1] + dbuf[2] + dbuf[3])
			e -= (mass[i] * mass[j]) / dist
		}
	}
	return e
}

// rsqrtBatch computes refined 1/sqrt for 4 values using RSQRTPS + Goldschmidt
func rsqrtBatch(r0, r1, r2, r3, c0375, c125, c1875, dtv archsimd.Float64x4) archsimd.Float64x4 {
	x0 := r0.Mul(r0)
	x1 := r1.Mul(r1)
	x2 := r2.Mul(r2)
	x3 := r3.Mul(r3)
	t0 := x0.AddPairsGrouped(x1)
	t1 := x2.AddPairsGrouped(x3)
	y0 := t0.Select128FromPair(1, 2, t1)
	y1 := t0.Select128FromPair(0, 3, t1)
	dsq := y0.Add(y1)
	approx := dsq.ConvertToFloat32().ReciprocalSqrt().ConvertToFloat64()
	yy := dsq.Mul(approx).Mul(approx)
	aa := yy.Mul(c0375).Mul(yy)
	bb := yy.Mul(c125).Sub(c1875)
	refined := approx.Mul(aa.Sub(bb))
	return refined.Mul(refined).Mul(refined).Mul(dtv)
}

func advance(steps int, dt float64) {
	dtv := archsimd.BroadcastFloat64x4(dt)
	c0375 := archsimd.BroadcastFloat64x4(0.375)
	c125 := archsimd.BroadcastFloat64x4(1.25)
	c1875 := archsimd.BroadcastFloat64x4(1.875)
	one := archsimd.BroadcastFloat64x4(1.0)

	rm0 := archsimd.BroadcastFloat64x4(mass[0])
	rm1 := archsimd.BroadcastFloat64x4(mass[1])
	rm2 := archsimd.BroadcastFloat64x4(mass[2])
	rm3 := archsimd.BroadcastFloat64x4(mass[3])
	rm4 := archsimd.BroadcastFloat64x4(mass[4])

	for s := 0; s < steps; s++ {
		// Pair deltas: (1,0),(2,0),(2,1),(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(4,3)
		r0 := pos[1].Sub(pos[0])
		r1 := pos[2].Sub(pos[0])
		r2 := pos[2].Sub(pos[1])
		r3 := pos[3].Sub(pos[0])

		// Batch 0-3: compute mag
		mag03 := rsqrtBatch(r0, r1, r2, r3, c0375, c125, c1875, dtv)
		var m03 [4]float64
		mag03.Store(&m03)

		r4 := pos[3].Sub(pos[1])
		r5 := pos[3].Sub(pos[2])
		r6 := pos[4].Sub(pos[0])
		r7 := pos[4].Sub(pos[1])

		// Batch 4-7
		mag47 := rsqrtBatch(r4, r5, r6, r7, c0375, c125, c1875, dtv)
		var m47 [4]float64
		mag47.Store(&m47)

		r8 := pos[4].Sub(pos[2])
		r9 := pos[4].Sub(pos[3])

		// Batch 8-9 (padded)
		mag89 := rsqrtBatch(r8, r9, one, one, c0375, c125, c1875, dtv)
		var m89 [4]float64
		mag89.Store(&m89)

		// Velocity updates - fully unrolled
		var t archsimd.Float64x4

		t = r0.Mul(archsimd.BroadcastFloat64x4(m03[0]))
		vel[1] = vel[1].Sub(t.Mul(rm0))
		vel[0] = vel[0].Add(t.Mul(rm1))

		t = r1.Mul(archsimd.BroadcastFloat64x4(m03[1]))
		vel[2] = vel[2].Sub(t.Mul(rm0))
		vel[0] = vel[0].Add(t.Mul(rm2))

		t = r2.Mul(archsimd.BroadcastFloat64x4(m03[2]))
		vel[2] = vel[2].Sub(t.Mul(rm1))
		vel[1] = vel[1].Add(t.Mul(rm2))

		t = r3.Mul(archsimd.BroadcastFloat64x4(m03[3]))
		vel[3] = vel[3].Sub(t.Mul(rm0))
		vel[0] = vel[0].Add(t.Mul(rm3))

		t = r4.Mul(archsimd.BroadcastFloat64x4(m47[0]))
		vel[3] = vel[3].Sub(t.Mul(rm1))
		vel[1] = vel[1].Add(t.Mul(rm3))

		t = r5.Mul(archsimd.BroadcastFloat64x4(m47[1]))
		vel[3] = vel[3].Sub(t.Mul(rm2))
		vel[2] = vel[2].Add(t.Mul(rm3))

		t = r6.Mul(archsimd.BroadcastFloat64x4(m47[2]))
		vel[4] = vel[4].Sub(t.Mul(rm0))
		vel[0] = vel[0].Add(t.Mul(rm4))

		t = r7.Mul(archsimd.BroadcastFloat64x4(m47[3]))
		vel[4] = vel[4].Sub(t.Mul(rm1))
		vel[1] = vel[1].Add(t.Mul(rm4))

		t = r8.Mul(archsimd.BroadcastFloat64x4(m89[0]))
		vel[4] = vel[4].Sub(t.Mul(rm2))
		vel[2] = vel[2].Add(t.Mul(rm4))

		t = r9.Mul(archsimd.BroadcastFloat64x4(m89[1]))
		vel[4] = vel[4].Sub(t.Mul(rm3))
		vel[3] = vel[3].Add(t.Mul(rm4))

		// Update positions
		pos[0] = pos[0].Add(vel[0].Mul(dtv))
		pos[1] = pos[1].Add(vel[1].Mul(dtv))
		pos[2] = pos[2].Add(vel[2].Mul(dtv))
		pos[3] = pos[3].Add(vel[3].Mul(dtv))
		pos[4] = pos[4].Add(vel[4].Mul(dtv))
	}
}

func main() {
	flag.Parse()
	if flag.NArg() > 0 {
		n, _ = strconv.Atoi(flag.Arg(0))
	}
	initBodies()
	offsetMomentum()
	fmt.Printf("%.9f\n", energy())
	advance(n, 0.01)
	fmt.Printf("%.9f\n", energy())
}
