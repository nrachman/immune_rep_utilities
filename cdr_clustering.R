assign_cdr_clusters_faster <- function(unique_cdr3, pcnt_identity_threshold, length_diff_threshold = 2){
  
  unique_cdr3_dat <- data.frame(cdr = unique_cdr3, len = nchar(unique_cdr3)) %>% arrange(len)

  min_length <- min(unique_cdr3_dat$len)
  max_length <- max(unique_cdr3_dat$len)
  length_windows_start <- seq(min_length, max_length - length_diff_threshold)
  length_windows_end <- length_windows_start + length_diff_threshold

  edge_list <- c()
  for(i in seq_along(length_windows_start)){
    print(paste("window start", length_windows_start[i]))
    seq_to_compare <- unique_cdr3_dat %>% filter(len %in% length_windows_start[i]:length_windows_end[i]) %>% pull(cdr)
    adjacency_mat <- get_pcnt_ident(seq_to_compare)

    names_mat <- outer(seq_to_compare, seq_to_compare, FUN = paste)
    names_vec <- as.vector(names_mat)
    adjacency_vec <- as.vector(adjacency_mat)
    edge_list <- c(edge_list, names_vec[adjacency_vec > pcnt_identity_threshold])

  }
  edge_list <- unique(edge_list)

  edge_list <- str_split(edge_list, " ")
  edge_dat <- matrix(ncol = 2, nrow = length(edge_list))
  edge_dat[,1] <- sapply(edge_list, `[[`, 1)
  edge_dat[,2] <- sapply(edge_list, `[[`, 2)
  #print(str(edge_dat))

  library(igraph)
  
  g <- graph_from_edgelist(edge_dat)
  enframe(components(g)$membership, "cdr", "membership")
}

get_pcnt_ident <- function(unique_cdr3){
  edit_dist_mat <- adist(unique_cdr3, counts = TRUE)

  x <- attr(edit_dist_mat, "trafos")
  out <- apply(x, 2, str_count, "M") / nchar(x)
  colnames(out) <- colnames(edit_dist_mat)
  rownames(out) <- rownames(edit_dist_mat)
  out
}

assign_cdr_clusters <- function(unique_cdr3, pcnt_identity_threshold){
  library(igraph)
  
  pcnt_ident_mat <- get_pcnt_ident(unique_cdr3)
  
  pcnt_ident_mat_thresh <- (pcnt_ident_mat > pcnt_identity_threshold) *1
  diag(pcnt_ident_mat_thresh) <- 0
  
  g <- graph_from_adjacency_matrix(pcnt_ident_mat_thresh, mode = "undirected")
  comp <- components(g)$membership
  data.frame(cdr = unique_cdr3, membership = comp)

}

