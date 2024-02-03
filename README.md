# llama-inference

---

Llama-inference is a simple implementation of Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) as a Go net/http webserver. 
The backbone of this project is really rooted around using [go-llama2](https://github.com/tmc/go-llama2). This for now is just a fun weekend project to continue to learn the MLOps side of things and break away from using Python as a backend for ML projects.

### Setup & Prerequisites For Running Locally

Before setting up the `llama-inference` project, ensure you have the following installed on your system:

- **Go (Golang)**: The project is developed in Go. You need to have Go installed to compile and run the server. Check if Go is installed by running `go version` in your terminal. If you need to install Go, follow the [official installation instructions](https://golang.org/doc/install).

- **Git**: Git is used to clone the project repository from GitHub. Check if Git is installed by running `git --version` in your terminal. If you need to install Git, follow the [official Git installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Installation

1. **Clone the Repository**: Clone the `llama-inference` repository to your local machine using Git:

    ```sh
    git clone https://github.com/tkruer/llama-infrence.git
    ```

    Navigate into the project directory:

    ```sh
    cd llama-infrence
    ```

2. **Directory Structure**: Ensure the directory structure looks like this after cloning:

    ```
    .
    ├── Dockerfile
    ├── README.md
    ├── go.mod
    ├── inference
    │   └── inference.go
    ├── main.go
    ├── model
    │   ├── stories15M.bin
    │   └── tokenizer.bin
    └── public
        └── views
            └── index.html
    ```

3. **Running the Server**: From the root of the project directory, run the Go server using:

    ```sh
    go run main.go
    ```

    This command compiles and starts the web server. By default, the server will listen on `http://localhost:10000`. You can open this URL in your web browser to interact with the application.


