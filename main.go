package main

import (
	"fmt"
	"github.com/tkruer/llama-inference/inference"
	"net/http"
	"strings"
)

func RemoveTagFromString(s string) string {
	return strings.ReplaceAll(s, "<s>", "<br>")
}

type requestBody struct {
	Message string `json:"message"`
}

func messageHandler(w http.ResponseWriter, r *http.Request) {

	if r.Method != "POST" {
		http.Error(w, "Method is not supported.", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseForm()
	if err != nil {
		http.Error(w, "Error parsing form data", http.StatusInternalServerError)
		return
	}

	message := r.FormValue("message")

	fmt.Printf("Received message: %s\n", message)

	result := inference.Inference(message, "./model/stories15M.bin", 1.0, 256)
	if err != nil {
		http.Error(w, "Error generating text", http.StatusInternalServerError)
		return
	}

	resultClean := RemoveTagFromString(result)

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(resultClean))
}

func main() {

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "./public/views/index.html")
	})

	http.HandleFunc("/api/message", messageHandler)

	fmt.Println("Server starting on http://localhost:10000")
	if err := http.ListenAndServe("0.0.0.0:10000", nil); err != nil {
		panic(err)
	}
}
