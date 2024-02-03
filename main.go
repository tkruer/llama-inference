package main

import (
	"encoding/json"
	"fmt"
	"github.com/tkruer/llama-inference/inference"
	"io"
	"net/http"
)

type requestBody struct {
	Message string `json:"message"`
}

func messageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method is not supported.", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}
	defer r.Body.Close()

	var msg requestBody
	err = json.Unmarshal(body, &msg)
	if err != nil {
		http.Error(w, "Error parsing request body", http.StatusBadRequest)
		return
	}

	fmt.Printf("Received message: %s\n", msg.Message)

	result := inference.Inference(msg.Message, "./model/stories15M.bin", 1.0, 256)
	if err != nil {
		http.Error(w, "Error generating text", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(result))
}

func main() {

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")

		w.Write([]byte("<p>Wrong way, use route <code>/message</code> instead</p>"))
	})

	http.HandleFunc("/message", messageHandler)

	fmt.Println("Server starting on http://localhost:8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		panic(err)
	}
}
