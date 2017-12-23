package main
import (
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"github.com/james-bowman/nlp"
	"io"
	"os"
	"bufio"
	)

type comment struct{ //not in use
	ID string
	text string
	targets []string
}

func main() {

	testCorpus := make([]string, 20)
    f, _ := os.Open("C:\\Users\\chent\\go\\src\\toxicComment\\testingCSVFile.csv")

	r := csv.NewReader(bufio.NewReader(f))
	//var comments = make([]comment, 0)
    for {
        record, err := r.Read()
        // Stop at EOF.
        if err == io.EOF {
            break
        }
        // Display record.
        // ... Display record length.
        // ... Display all individual elements of the slice.
        //comments = append(comments, comment{record[0],record[1],record[2:]})
        testCorpus = append(testCorpus, record[1])
    }
	query := "I don't think a stuffed arm really counts as an appearance."

	vectoriser := nlp.NewCountVectoriser(true)
	transformer := nlp.NewTfidfTransformer()

	// set k (the number of dimensions following truncation) to 4
	reducer := nlp.NewTruncatedSVD(4)

	// Transform the corpus into an LSI fitting the model to the documents in the process
    fmt.Println("Vectorise the corpus into an LSI fitting the model to the documents in the process")
	matrix, _ := vectoriser.FitTransform(testCorpus...)
    fmt.Println("Transform the corpus into an LSI fitting the model to the documents in the process")
	matrix, _ = transformer.FitTransform(matrix)
    fmt.Println("Reduce the corpus into an LSI fitting the model to the documents in the process")

    /*
		This is where the error starts
	*/
	lsi, _ := reducer.FitTransform(matrix)

	// run the query through the same pipeline that was fitted to the corpus and
	// to project it into the same dimensional space
	fmt.Println("run the query through the same pipeline that was fitted to the corpus")
	matrix, _ = vectoriser.Transform(query)
	matrix, _ = transformer.Transform(matrix)
	queryVector, _ := reducer.Transform(matrix)

	// iterate over document feature vectors (columns) in the LSI and compare with the
	// query vector for similarity.  Similarity is determined by the difference between
	// the angles of the vectors known as the cosine similarity
	fmt.Println("iterate over document feature vectors (columns) in the LSI and compare")
	highestSimilarity := -1.0
	var matched int
	_, docs := lsi.Dims()
	for i := 0; i < docs; i++ {
		queryVec := queryVector.(mat.ColViewer).ColView(0)
		docVec := lsi.(mat.ColViewer).ColView(i)
		similarity := nlp.CosineSimilarity(queryVec, docVec)		
		if similarity > highestSimilarity {
			matched = i
			highestSimilarity = similarity
		}
	}

	fmt.Printf("Matched '%s'\n", testCorpus[matched])
	// Output: Matched 'The quick brown fox jumped over the lazy dog'
}