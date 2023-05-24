#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <limits.h>
#include <math.h>

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Structures Structures Structures Structures Structures Structures Structures Stru
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Structures Structures Structures Structures Structures Structures Structures Stru
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Structures Structures Structures Structures Structures Structures Structures Stru
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Queue - Definition of a queue structure needed for BFS and Brandes algorithms
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
typedef struct queue {
	int capacity;
	int front;
	int rear;
	int size;
	int* value;
} queue;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Queue Constructor - Initializes a queue of a given capacity
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct queue* initializeQueue(int capacity) {
	struct queue* q = (struct queue*)malloc(sizeof(struct queue));

	q->capacity = capacity;
	q->front = 0;
	q->rear = capacity - 1;
	q->size = 0;
	q->value = (int*)malloc(capacity * sizeof(int));

	return q;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Enqueue - Adds a node to the end of the queue
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void enqueue(struct queue* q, int node) {
	if (q->size == q->capacity) {
		printf("\nQueue is full, enqueue aborted\n");
		fflush(stdout);

		return;
	}

	q->rear = (q->rear + 1) % q->capacity;
	q->value[q->rear] = node;
	q->size = q->size + 1;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Dequeue - Returns the first item from the queue
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int dequeue(struct queue* q) {
	if (q->size == 0) {
		printf("\nQueue is empty, dequeue aborted\n");
		fflush(stdout);

		return -1;
	}

	int node = q->value[q->front];

	q->front = (q->front + 1) % q->capacity;
	q->size = q->size - 1;

	return node;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Stack - Definition of a stack structure needed for the Brandes algorithm
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
typedef struct stack {
	int capacity;
	int top;
	int* value;
} stack;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Stack Constructor - Initializes a stack of a given capacity
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct stack* initializeStack(int capacity) {
	struct stack* s = (struct stack*)malloc(sizeof(struct stack));

	s->capacity = capacity;
	s->top = 0;
	s->value = (int*)malloc(capacity * sizeof(int));

	return s;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Push - Adds an item to the top of the stack
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void push(struct stack* s, int node) {
	if (s->top == s->capacity) {
		printf("\n stack top = %d\n", s->top);
					fflush(stdout);
		printf("\nStack is full, push aborted\n");
		fflush(stdout);

		return;
	}

	s->value[s->top] = node;
	s->top = s->top + 1;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Pop - Returns the last item from the stack
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int pop(struct stack* s) {
	if (s->top == 0) {
		printf("\nStack is empty, pop aborted\n");
		fflush(stdout);

		return -1;
	}

	s->top = s->top - 1;

	int node = s->value[s->top];

	return node;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	List - Definition of a list structure through nodes
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
typedef struct node {
	int value;
	struct node* next;
} node;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	List Constructor - Initializes a list with a given value
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct node* initializeNode(int value) {
	struct node* l = (struct node*)malloc(sizeof(struct node));

	l->value = value;
	l->next = NULL;

	return l;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Add Node to List - Appends a new node to a list
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct node* addNode(struct node* n, int value) {
	struct node* newNode = initializeNode(value);
	struct node* temp;

	if (n == NULL) {
		n = newNode;
	} else {
		temp = n;
		while(temp->next != NULL) {
			temp = temp->next;
		}

		temp->next = newNode;
	}

	return n;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Graph - Definition of a graph structure through adjecency lists
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
typedef struct graph {
	int capacity;
	struct node** adjLists;
} graph;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	List Constructor - Initializes a graph of a given capacity
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct graph* initializeGraph(int capacity) {
	struct graph* g = (struct graph*)malloc(sizeof(struct graph));

	g->capacity = capacity;
	g->adjLists = malloc(capacity * sizeof(struct node*));

	for (int i = 0; i < capacity; i++)
		g->adjLists[i] = 0;

	return g;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Add Edge - Add an undirected from 2 vertices to the graph
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void addEdge(struct graph* g, int v, int w) {
	struct node* newNode = initializeNode(w);
	newNode->next = g->adjLists[v];
	g->adjLists[v] = newNode;

	newNode = initializeNode(v);
	newNode->next = g->adjLists[w];
	g->adjLists[w] = newNode;
}



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO IO
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Get Vertex Count - Returns n which is used throughout the application
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int getVertexCount(char* fileName) {
	FILE *f;
	char line[256];

	f = fopen(fileName, "r");
	fgets(line, sizeof(line), f);

	int n = atoi(line);

	return n;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Get Graph Dimensions - Returns the number of nodes and the largest degree
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void getGraphDimensions(char* fileName, char* delimiter, int *n, int *m) {
	FILE *f;
	char line[256];

	f = fopen(fileName, "r");
	fgets(line, sizeof(line), f);

	*n = atoi(line);

	int counter[*n];
	int from, to;

	for (int i = 0; i < *n; i++)
		counter[i] = 0;

	while(fgets(line, sizeof(line), f)) {
			char *token =  strtok(line, delimiter);
			from = strtol(token, NULL, 10);

			token =  strtok(NULL, delimiter);
			to = strtol(token, NULL, 10);

			counter[from]++;
			counter[to]++;
		}

	int max = 0;

	for (int i = 0; i < *n; i++) {
		if (max < counter[i])
			max = counter[i];
	}

	*m = max;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Graph Loader - Initializes a graph from a file, through an adjacency matrix
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void loadGraph(char* fileName, char* delimiter, int* graph) {
	FILE *f;
	int from, to;
	char line[256];

	f = fopen(fileName, "r");
	fgets(line, sizeof(line), f);
	int n = atoi(line);

	for (int i = 0; i < n * n; i++)
		graph[i] = 0;

	while(fgets(line, sizeof(line), f)) {
		char *token =  strtok(line, delimiter);
		from = strtol(token, NULL, 10);

		token =  strtok(NULL, delimiter);
		to = strtol(token, NULL, 10);

		graph[from * n + to] = 1;
		graph[to * n + from] = 1;
	}

	printf("Graph loaded successfully\n");
	fflush(stdout);

	fclose(f);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Graph Loader - Initializes a graph and counter from a file to make an adj. list
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void loadGraphMatList(char* fileName, char* delimiter, int n, int m,
						int* graph, int* counter) {
	FILE *f;
	int from, to;
	char line[256];

	f = fopen(fileName, "r");
	fgets(line, sizeof(line), f);

	for (int i = 0; i < n * m; i++)
		graph[i] = 0;

	for (int i = 0; i < n; i++)
		counter[i] = 0;

	while(fgets(line, sizeof(line), f)) {
		char *token =  strtok(line, delimiter);
		from = strtol(token, NULL, 10);

		token =  strtok(NULL, delimiter);
		to = strtol(token, NULL, 10);

		graph[from * m + counter[from]] = to;
		graph[to * m + counter[to]] = from;

		counter[from]++;
		counter[to]++;
	}

	printf("Graph loaded successfully\n");
	fflush(stdout);

	fclose(f);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Graph Object Loader - Initializes a graph from a file, through adjacency lists
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void loadGraphObject(char* fileName, char* delimiter, struct graph* g) {
	FILE *f;
	int from, to;
	char line[256];

	f = fopen(fileName, "r");
	fgets(line, sizeof(line), f);

	while(fgets(line, sizeof(line), f)) {
		if (strcmp("from,to\n", line) != 0) {
			char *token =  strtok(line, delimiter);
			from = strtol(token, NULL, 10);

			token =  strtok(NULL, delimiter);
			to = strtol(token, NULL, 10);

			addEdge(g, from, to);
		}
	}

	printf("Graph loaded successfully\n");
	fflush(stdout);

	fclose(f);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Write Result - Writes the final result array to a file
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void writeResult(int n, char* fileName, double* result) {
	FILE *f;

	f = fopen(fileName, "w");

	for (int i = 0; i < n; i++)
		fprintf(f,"%.4f\n", result[i]);

	fclose(f);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Closeness Closeness Closeness Closeness Closeness Closeness Closeness Closeness Cl
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Closeness Closeness Closeness Closeness Closeness Closeness Closeness Closeness Cl
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Closeness Closeness Closeness Closeness Closeness Closeness Closeness Closeness Cl
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Closeness Algorithm - Returns the closeness centrality for a node
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double closenessAlgorithmList(int n, struct graph* g, int source) {
	int *distance = malloc(n * sizeof(int));

	for (int i = 0; i < n; i++) {
		distance[i] = -1;
	}

	distance[source] = 0;

	struct queue* q = initializeQueue(n);
	enqueue(q, source);

	while(q->size != 0) {
		int v = dequeue(q);

		struct node* temp = g->adjLists[v];

		while(temp != NULL) {
			int w = temp->value;

			if (distance[w] < 0) {
				enqueue(q, w);
				distance[w] = distance[v] + 1;
			}

			temp = temp->next;
		}
	}

	double farness = 0;

	for (int i = 0; i < n; i++) {
		farness += distance[i];
	}

	free(distance);

	return (n - 1) / farness;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Closeness Algorithm - Adj. Matrix version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double closenessAlgorithm(int n, int *graph, int source) {
	int *distance = malloc(n * sizeof(int));

	for (int i = 0; i < n; i++) {
		distance[i] = -1;
	}

	distance[source] = 0;

	struct queue* q = initializeQueue(n);
	enqueue(q, source);

	while(q->size != 0) {
		int v = dequeue(q);

		for (int w = 0; w < n; w++) {
			if (v != w && graph[n * v + w] != 0) {
				if (distance[w] < 0) {
					enqueue(q, w);
					distance[w] = distance[v] + 1;
				}
			}
		}
	}

	double farness = 0;

	for (int i = 0; i < n; i++) {
		farness += distance[i];
	}

	free(distance);

	return (n - 1) / farness;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Closeness Algorithm - Adj. Matrix List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double closenessAlgorithmMatList(int n, int m, int *graph, int *counter,
									int source) {
	int *distance = malloc(n * sizeof(int));

	for (int i = 0; i < n; i++) {
		distance[i] = -1;
	}

	distance[source] = 0;

	struct queue* q = initializeQueue(n);
	enqueue(q, source);

	while(q->size != 0) {
		int v = dequeue(q);

		for (int i = 0; i < counter[v]; i++) {
			int w = graph[v * m + i];

			if (distance[w] < 0) {
				enqueue(q, w);
				distance[w] = distance[v] + 1;
			}
		}
	}

	double farness = 0;

	for (int i = 0; i < n; i++) {
		farness += distance[i];
	}

	free(distance);

	return (n - 1) / farness;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Closeness Centrality - Sequential Adjacency List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateClosenessList(int n, struct graph* g, double *cc) {
	for (int i = 0; i < n; i++)
		cc[i] = closenessAlgorithmList(n, g, i);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Closeness Centrality - Parallel Adjacency Matrix version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateCloseness(int n, int *graph, double *cc, int p,
									int my_rank, int local_n) {
	MPI_Bcast(graph, n * n, MPI_INT, 0, MPI_COMM_WORLD);

	double *local_cc = malloc(n * sizeof(double));

	for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++)
		local_cc[i] = closenessAlgorithm(n, graph, i);

	MPI_Reduce(local_cc, cc, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) {
		int leftover_n = n - local_n * (n / p);

		if (leftover_n != 0) {
			for (int i = local_n * p; i < n; i++)
				cc[i] = closenessAlgorithm(n, graph, i);
		}
	}

	free(local_cc);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Closeness Centrality - Parallel Adjacency Matrix List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateClosenessMatList(int n, int m, int *graph, int *counter, double *cc,
									int p, int my_rank, int local_n) {
	MPI_Bcast(graph, n * m, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(counter, n, MPI_INT, 0, MPI_COMM_WORLD);

	double *local_cc = malloc(n * sizeof(double));

	for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++)
		local_cc[i] = closenessAlgorithmMatList(n, m, graph, counter, i);

	MPI_Reduce(local_cc, cc, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) {
		int leftover_n = n - local_n * (n / p);

		if (leftover_n != 0) {
			for (int i = local_n * p; i < n; i++)
				cc[i] = closenessAlgorithmMatList(n, m, graph, counter, i);
		}
	}

	free(local_cc);
}



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Betweenness Betweenness Betweenness Betweenness Betweenness Betweenness Betweenne
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Betweenness Betweenness Betweenness Betweenness Betweenness Betweenness Betweenne
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Betweenness Betweenness Betweenness Betweenness Betweenness Betweenness Betweenne
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Brandes algorithm - Returns the Betweenness Centrality for a given node
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void brandesAlgorithmList(int n, struct graph* g, double *bc, int source) {
	double *counter = malloc(n * sizeof(double));
	int *distance = malloc(n * sizeof(int));
	double *delta = malloc(n * sizeof(double));
	struct graph* paths = initializeGraph(n);

	for (int i = 0; i < n; i++) {
		counter[i] = 0;
		distance[i] = -1;
		delta[i] = 0;
	}

	counter[source] = 1;
	distance[source] = 0;

	struct stack* s = initializeStack(n);

	struct queue* q = initializeQueue(n);
	enqueue(q, source);

	struct node* temp;

	while(q->size != 0) {
		int v = dequeue(q);
		push(s, v);

		temp = g->adjLists[v];

		while(temp != NULL) {
			int w = temp->value;

			if (distance[w] < 0) {
				enqueue(q, w);
				distance[w] = distance[v] + 1;
			}

			if (distance[w] == distance[v] + 1) {
				counter[w] = counter[w] + counter[v];
				paths->adjLists[w] = addNode(paths->adjLists[w], v);
			}

			temp = temp->next;
		}
	}

	while(s->top != 0) {
		int w = pop(s);

		temp = paths->adjLists[w];

		while (temp != NULL) {
			int v = temp->value;

			delta[v] += (counter[v] / counter[w]) * (1 + delta[w]);

			temp = temp->next;
		}

		if (w != source) {
			bc[w] += delta[w];
		}
	}

	free(counter);
	free(distance);
	free(paths);
	free(delta);
	free(temp);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Brandes algorithm - Adj. Matrix version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void brandesAlgorithm(int n, int *graph, double *bc, int source) {
	double *counter = malloc(n * sizeof(double));
	int *distance = malloc(n * sizeof(int));
	int *paths = malloc(n * n * sizeof(int));
	int *path_counter = malloc(n * sizeof(int));
	double *delta = malloc(n * sizeof(double));

	for (int i = 0; i < n; i++) {
		counter[i] = 0;
		distance[i] = -1;
		path_counter[i] = 0;
		delta[i] = 0;
	}

	for (int i = 0; i < n * n; i++)
		paths[i] = 0;

	counter[source] = 1;
	distance[source] = 0;

	struct stack* s = initializeStack(n);

	struct queue* q = initializeQueue(n);
	enqueue(q, source);

	while(q->size != 0) {
		int v = dequeue(q);
		push(s, v);

		for (int w = 0; w < n; w++) {
			if (v != w && graph[n * v + w] != 0) {
				if (distance[w] < 0) {
					enqueue(q, w);
					distance[w] = distance[v] + 1;
				}

				if (distance[w] == distance[v] + 1) {
					counter[w] = counter[w] + counter[v];
					paths[w * n + path_counter[w]] = v;
					path_counter[w]++;
				}
			}
		}
	}

	while(s->top != 0) {
		int w = pop(s);

		for (int i = 0; i < path_counter[w]; i++) {
			int v = paths[w * n + i];

			delta[v] += (counter[v] / counter[w]) * (1 + delta[w]);
		}

		if (w != source) {
			bc[w] += delta[w];
		}
	}

	free(counter);
	free(distance);
	free(paths);
	free(path_counter);
	free(delta);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Brandes algorithm - Adj. Matrix List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void brandesAlgorithmMatList(int n, int m, int *graph, int *graphCounter,
								double *bc, int source) {
	double *counter = malloc(n * sizeof(double));
	int *distance = malloc(n * sizeof(int));
	int *paths = malloc(n * n * sizeof(int));
	int *path_counter = malloc(n * sizeof(int));
	double *delta = malloc(n * sizeof(double));

	for (int i = 0; i < n; i++) {
		counter[i] = 0;
		distance[i] = -1;
		path_counter[i] = 0;
		delta[i] = 0;
	}

	for (int i = 0; i < n * n; i++)
		paths[i] = 0;

	counter[source] = 1;
	distance[source] = 0;

	struct stack* s = initializeStack(n);

	struct queue* q = initializeQueue(n);
	enqueue(q, source);

	while(q->size != 0) {
		int v = dequeue(q);
		push(s, v);

		for (int i = 0; i < graphCounter[v]; i++) {
			int w = graph[v * m + i];

			if (distance[w] < 0) {
				enqueue(q, w);
				distance[w] = distance[v] + 1;
			}

			if (distance[w] == distance[v] + 1) {
				counter[w] = counter[w] + counter[v];
				paths[w * n + path_counter[w]] = v;
				path_counter[w]++;
			}
		}
	}

	while(s->top != 0) {
		int w = pop(s);

		for (int i = 0; i < path_counter[w]; i++) {
			int v = paths[w * n + i];

			delta[v] += (counter[v] / counter[w]) * (1 + delta[w]);
		}

		if (w != source) {
			bc[w] += delta[w];
		}
	}

	free(counter);
	free(distance);
	free(paths);
	free(path_counter);
	free(delta);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Betweenness Centrality - Sequential Adjacency List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateBetweennessList(int n, struct graph* g, double *bc) {
	for (int i = 0; i < n; i++)
		bc[i] = 0;

	for (int i = 0; i < n; i++)
		brandesAlgorithmList(n, g, bc, i);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Betweenness Centrality - Parallel Adjacency Matrix version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateBetwenness(int n, int *graph, double *bc, int p, int my_rank,
							int local_n) {
	MPI_Bcast(graph, n * n, MPI_INT, 0, MPI_COMM_WORLD);

	double *local_bc = malloc(n * sizeof(double));

	for (int i = 0; i < n; i++)
			local_bc[i] = 0;

	for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++)
		brandesAlgorithm(n, graph, local_bc, i);

	MPI_Reduce(local_bc, bc, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) {
		int leftover_n = n - local_n * (n / p);

		if (leftover_n != 0) {
			for (int i = local_n * p; i < n; i++)
				brandesAlgorithm(n, graph, local_bc, i);
		}
	}

	free(local_bc);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Betweenness Centrality - Parallel Adjacency Matrix List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateBetwennessMatList(int n, int m, int *graph, int *counter, double *bc,
								int p, int my_rank, int local_n) {
	MPI_Bcast(graph, n * m, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(counter, n, MPI_INT, 0, MPI_COMM_WORLD);

	double *local_bc = malloc(n * sizeof(double));

	for (int i = 0; i < n; i++)
		local_bc[i] = 0;

	for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++)
		brandesAlgorithmMatList(n, m, graph, counter, local_bc, i);

	MPI_Reduce(local_bc, bc, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) {
		int leftover_n = n - local_n * (n / p);

		if (leftover_n != 0) {
			for (int i = local_n * p; i < n; i++)
				brandesAlgorithmMatList(n, m, graph, counter, bc, i);
		}
	}

	free(local_bc);
}



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Eigenvector Eigenvector Eigenvector Eigenvector Eigenvector Eigenvector Eigenvecto
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Eigenvector Eigenvector Eigenvector Eigenvector Eigenvector Eigenvector Eigenvecto
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Eigenvector Eigenvector Eigenvector Eigenvector Eigenvector Eigenvector Eigenvecto
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Get Eigenvalue - Finds the largest eigenvalue from an adj. Matrix
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double eigenvalueAlgorithmList(int n, struct graph* g, int iteration,
								double *eigenvector) {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Power Iteration - Returns largest eigenvector
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	for (int i = 0; i < n; i++)
		eigenvector[i] = 1;

	double *temp = malloc(n * sizeof(double));
	double *matrix = malloc(n * sizeof(double));

	struct node* node;

	for (int i = 0; i < iteration; i++) {
		for (int i = 0; i < n; i++) {
			temp[i] = 0;

			node = g->adjLists[i];

			while (node != NULL) {
				int v = node->value;

				temp[i] += eigenvector[v];

				node = node->next;
			}
		}

		for (int i = 0; i < n; i++)
			eigenvector[i] = temp[i];

		double norm = 0;

		for (int i = 0; i < n; i++)
			norm += eigenvector[i] * eigenvector[i];

		norm = sqrt(norm);

		for (int i = 0; i < n; i++)
			eigenvector[i] = eigenvector[i] / norm;
	}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Rayleigh quotient - Finds eigenvalue from eigenvector
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	double divider = 0;
	double result = 0;


	for (int i = 0; i < n; i++) {
		matrix[i] = 0;
		divider += eigenvector[i] * eigenvector[i];
	}


	for (int i = 0; i < n; i++) {
		node = g->adjLists[i];

		while (node != NULL) {
			int v = node->value;

			matrix[i] += eigenvector[v];

			node = node->next;
		}

		result += eigenvector[i] * matrix[i];
	}

	result = result / divider;

	free(temp);
	free(matrix);

	return result;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Get Eigenvalue - Finds the largest eigenvalue from an adj. Matrix
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double eigenvalueAlgorithm(int n, int *graph, int iteration,
									double *eigenvector, int p,
									int my_rank, int local_n) {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Power Iteration - Returns largest eigenvector
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	double *temp = malloc(n * sizeof(double));
	double *matrix = malloc(n * sizeof(double));

	int leftover_n = n - local_n * (n / p);

	if (my_rank == 0) {
		for (int i = 0; i < n; i++)
			eigenvector[i] = 1;
	}

	for (int i = 0; i < iteration; i++) {
		MPI_Bcast(eigenvector, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++) {
			temp[i] = 0;

			for (int j = 0; j < n; j++)
				temp[i] += graph[i * n + j] * eigenvector[j];
		}

		if (my_rank == 0) {
			for(int proc = 1; proc < p; ++proc)
				MPI_Recv(eigenvector + proc * local_n, local_n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else
			MPI_Send(temp + my_rank * local_n, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

		if (my_rank == 0) {
			if (leftover_n != 0) {
				for (int i = local_n * p; i < n; i++) {
					temp[i] = 0;

					for (int j = 0; j < n; j++)
						temp[i] += graph[i * n + j] * eigenvector[j];
				}
			}

			for (int i = 0; i < n; i++)
				eigenvector[i] = temp[i];

			double norm = 0;

			for (int i = 0; i < n; i++)
				norm += eigenvector[i] * eigenvector[i];

			norm = sqrt(norm);

			for (int i = 0; i < n; i++)
				eigenvector[i] = eigenvector[i] / norm;
		}
	}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Rayleigh quotient - Finds eigenvalue from eigenvector
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	MPI_Bcast(eigenvector, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double divider = 0;
	double result = 0;
	double local_result = 0;

	if (my_rank == 0) {
		for (int i = 0; i < n; i++) {
			matrix[i] = 0;
			divider += eigenvector[i] * eigenvector[i];
		}
	} else {
		for (int i = 0; i < n; i++)
			matrix[i] = 0;
	}

	for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++) {
		for (int j = 0; j < n; j++)
			matrix[i] += graph[i * n + j] * eigenvector[j];

		local_result += eigenvector[i] * matrix[i];
	}

	MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) {
		if (leftover_n != 0) {
			for (int i = local_n * p; i < n; i++) {
				for (int j = 0; j < n; j++)
					matrix[i] += graph[i * n + j] * eigenvector[j];

				result += eigenvector[i] * matrix[i];
			}
		}

		result = result / divider;
	}

	MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(temp);
	free(matrix);

	return result;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Get Eigenvalue - Adj. Matrix List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double eigenvalueAlgorithmMatList(int n, int m, int *graph, int *counter,
								int iteration, double *eigenvector, int p,
								int my_rank, int local_n) {
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Power Iteration - Returns largest eigenvector
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	double *temp = malloc(n * sizeof(double));
	double *local_temp = malloc(n * sizeof(double));
	double *matrix = malloc(n * sizeof(double));

	int leftover_n = n - local_n * (n / p);

	if (my_rank == 0) {
		for (int i = 0; i < n; i++)
			eigenvector[i] = 1;
	}

	for (int it = 0; it < iteration; it++) {
		MPI_Bcast(eigenvector, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++) {
			local_temp[i] = 0;

			for (int j = 0; j < counter[i]; j++) {
				int v = graph[i * m + j];
				local_temp[i] += eigenvector[v];
			}
		}

		MPI_Reduce(local_temp, temp, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		if (my_rank == 0) {
			if (leftover_n != 0) {
				for (int i = local_n * p; i < n; i++) {
					temp[i] = 0;

					for (int j = 0; j < counter[i]; j++) {
						int v = graph[i * m + j];
						temp[i] += eigenvector[v];
					}
				}
			}

			for (int i = 0; i < n; i++)
				eigenvector[i] = temp[i];

			double norm = 0;

			for (int i = 0; i < n; i++)
				norm += eigenvector[i] * eigenvector[i];

			norm = sqrt(norm);

			for (int i = 0; i < n; i++)
				eigenvector[i] = eigenvector[i] / norm;
		}
	}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Rayleigh quotient - Finds eigenvalue from eigenvector
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	MPI_Bcast(eigenvector, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double divider = 0;
	double result = 0;
	double local_result = 0;

	for (int i = 0; i < n; i++) {
		matrix[i] = 0;
		divider += eigenvector[i] * eigenvector[i];
	}

	for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++) {
		for (int j = 0; j < counter[i]; j++) {
			int v = graph[i * m + j];
			matrix[i] += eigenvector[v];
		}

		local_result += eigenvector[i] * matrix[i];
	}

	MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (my_rank == 0) {
		if (leftover_n != 0) {
			for (int i = local_n * p; i < n; i++) {
				for (int j = 0; j < counter[i]; j++) {
					int v = graph[i * m + j];
					matrix[i] += eigenvector[v];
				}

				result += eigenvector[i] * matrix[i];
			}
		}

		result = result / divider;
	}

	MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(temp);
	free(local_temp);
	free(matrix);

	return result;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Eigenvector Algorithm - Returns eigenvector centralty for largest eigenvalue
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void eigenvectorAlgorithmList(int n, struct graph* g, int iteration,
							double eigenvalue, double *result) {
	double *temp = malloc(n * sizeof(double));
	struct node* node;

	for (int i = 0; i < n; i++)
		result[i] = 1;

	for (int it = 0; it < iteration; it++) {
		for (int i = 0; i < n; i++) {
			temp[i] = 0;

			node = g->adjLists[i];

			while (node != NULL) {
				int v = node->value;

				temp[i] += result[v];

				node = node->next;
			}
		}

		for (int i = 0; i < n; i++)
			result[i] = temp[i] / eigenvalue;
	}

	free(temp);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Eigenvector Algorithm - Adj. Matrix version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void eigenvectorAlgorithm(int n, int *graph, int iteration,
									double eigenvalue, double *result,
									int p, int my_rank, int local_n) {
	double *temp = malloc(local_n * sizeof(double));

	if (my_rank == 0) {
		for (int i = 0; i < n; i++)
			result[i] = 1;
	}

	for (int it = 0; it < iteration; it++) {
		MPI_Bcast(result, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		for (int i = local_n * my_rank; i < local_n * (my_rank + 1); i++) {
			temp[i] = 0;

			for (int j = 0; j < n; j++)
				temp[i] += graph[i * n + j] * result[j];
		}

		if (my_rank == 0) {
			for(int proc = 1; proc < p; ++proc)
				MPI_Recv(temp + proc * local_n, local_n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else
			MPI_Send(temp + my_rank * local_n, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

		if (my_rank == 0) {
			int leftover_n = n - local_n * (n / p);

			if (leftover_n != 0) {
				for (int i = local_n * p; i < n; i++) {
					temp[i] = 0;

					for (int j = 0; j < n; j++)
						temp[i] += graph[i * n + j] * result[j];
				}
			}

			for (int i = 0; i < n; i++)
				result[i] = temp[i] / eigenvalue;
		}
	}

	free(temp);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Eigenvector Algorithm - Adj. Matrix List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void eigenvectorAlgorithmMatList(int n, int m, int *graph, int *counter,
							int iteration, double eigenvalue, double *result,
							int p, int my_rank, int local_n) {
	double *temp = malloc(n * sizeof(double));

	for (int i = 0; i < n; i++)
		result[i] = 1;

	for (int it = 0; it < iteration; it++) {
		MPI_Bcast(result, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		for (int i = 0; i < n; i++) {
			temp[i] = 0;

			for (int j = 0; j < counter[i]; j++) {
				int v = graph[i * m + j];
				temp[i] += result[v];
			}
		}

		if (my_rank == 0) {
			for(int proc = 1; proc < p; ++proc)
				MPI_Recv(temp + proc * local_n, local_n, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else
			MPI_Send(temp + my_rank * local_n, local_n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

		if (my_rank == 0) {
			int leftover_n = n - local_n * (n / p);

			if (leftover_n != 0) {
				for (int i = local_n * p; i < n; i++) {
					temp[i] = 0;

					for (int j = 0; j < counter[i]; j++) {
						int v = graph[i * m + j];
						temp[i] += result[v];
					}
				}
			}

			for (int i = 0; i < n; i++)
				result[i] = temp[i] / eigenvalue;
		}
	}

	free(temp);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Eigenvector Centrality - Sequential Adjacency List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateEigenvectorList(int n, struct graph* g, int iteration,
								double *result) {
	double *eigenvector = malloc(n * sizeof(double));

	double eigenvalue = eigenvalueAlgorithmList(n, g, iteration, eigenvector);

	eigenvectorAlgorithmList(n, g, iteration, eigenvalue, result);

	free(eigenvector);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Eigenvector Centrality - Parallel Adjacency Matrix version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateEigenvector(int n, int *graph, int iteration, double *result,
									int p, int my_rank, int local_n) {
	MPI_Bcast(graph, n * n, MPI_INT, 0, MPI_COMM_WORLD);

	double *eigenvector = malloc(n * sizeof(double));

	double eigenvalue = eigenvalueAlgorithm(n, graph, iteration, eigenvector, p,
												my_rank, local_n);

	eigenvectorAlgorithm(n, graph, iteration, eigenvalue, result, p, my_rank,
							local_n);

	free(eigenvector);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Eigenvector Centrality - Parallel Adjacency Matrix List version
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void calculateEigenvectorMatList(int n, int m, int *graph, int *counter,
									int iteration, double *result, int p,
									int my_rank, int local_n) {
	MPI_Bcast(graph, n * m, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(counter, n, MPI_INT, 0, MPI_COMM_WORLD);

	double *eigenvector = malloc(n * sizeof(double));

	double eigenvalue = eigenvalueAlgorithmMatList(n, m, graph, counter, iteration,
													eigenvector, p, my_rank,
													local_n);

	eigenvectorAlgorithmMatList(n, m, graph, counter, iteration, eigenvalue,
									result, p, my_rank, local_n);

	free(eigenvector);
}



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main M
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main M
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main Main M
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int main(int argc, char* argv[]){
	int  my_rank, p, n, m, local_n;

	char *fileName = "/home/clusteruser/SBMGraph.txt";
	char *closenessResultName = "/home/clusteruser/closeness_result.txt";
	char *betweennessResultName = "/home/clusteruser/betweenness_result.txt";
	char *eigenvectorResultName = "/home/clusteruser/eigenvector_result.txt";

	int type = 2; // 0 - closeness, 1 - betweenness, 2 - eigenvector
	int impl = 2; // 0 - adj. list, 1 - adj. matrix, 2 - adj. matrix list

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	double *result = malloc(n * sizeof(double));

	if (my_rank == 0) {
		if (impl == 2)
			getGraphDimensions(fileName, ",", &n, &m);
		else
			n = getVertexCount(fileName);
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (impl == 2)
		MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

	local_n = n / p;

	double t_begin = MPI_Wtime();

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency List implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency List implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency List implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	if (impl == 0) {
		struct graph* g = initializeGraph(n);

		if (my_rank == 0){
			loadGraphObject(fileName, ",", g);
		}

		if (type == 0) {
			calculateClosenessList(n, g, result);
			if (my_rank == 0) {
				writeResult(n, closenessResultName, result);
			}
		} else if (type == 1) {
			calculateBetweennessList(n, g, result);
			if (my_rank == 0) {
				writeResult(n, betweennessResultName, result);
			}
		} else {
			calculateEigenvectorList(n, g, 10, result);
			if (my_rank == 0) {
				writeResult(n, eigenvectorResultName, result);
			}
		}

		free(g);
	} else if (impl == 1) {

		int *graph = malloc(n * n * sizeof(int));;

		if (my_rank == 0)
			loadGraph(fileName, ",", graph);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency Matrix implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency Matrix implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency Matrix implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		if (type == 0) {
			calculateCloseness(n, graph, result, p, my_rank, local_n);
			if (my_rank == 0) {
				writeResult(n, closenessResultName, result);
			}
		} else if (type == 1) {
			calculateBetwenness(n, graph, result, p, my_rank, local_n);
			if (my_rank == 0) {
				writeResult(n, betweennessResultName, result);
			}
		} else {
			calculateEigenvector(n, graph, 10, result, p, my_rank, local_n);
			if (my_rank == 0) {
				writeResult(n, eigenvectorResultName, result);
			}
		}

		free(graph);

	} else {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency Matrix List implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency Matrix List implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//-----------------------------------------------------------------------------------
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	Adjacency Matrix List implementations
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		int *graph = malloc(n * m * sizeof(int));;
		int *counter = malloc(n * sizeof(int));

		if (my_rank == 0)
			loadGraphMatList(fileName, ",", n, m, graph, counter);

		if (type == 0) {
			calculateClosenessMatList(n, m, graph, counter, result, p, my_rank,
										local_n);
			if (my_rank == 0) {
				writeResult(n, closenessResultName, result);
			}
		} else if (type == 1) {
			calculateBetwennessMatList(n, m, graph, counter, result, p, my_rank,
										local_n);
			if (my_rank == 0) {
				writeResult(n, betweennessResultName, result);
			}
		} else {
			calculateEigenvectorMatList(n, m, graph, counter, 10, result, p,
											my_rank, local_n);
			if (my_rank == 0) {
				writeResult(n, eigenvectorResultName, result);
			}
		}

		free(graph);
		free(counter);
	}

	double t_end = MPI_Wtime();

	if (my_rank == 0) {
		printf("\nDuration of program: %.4f\n\n", t_end - t_begin);
		fflush(stdout);
	}

	free(result);

	MPI_Finalize();
	
	
	return 0;
}
