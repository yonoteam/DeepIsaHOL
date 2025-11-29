(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Generic frequently used ML operations.
*)

signature OPS =
sig
  (* Generic operations *)
  val ith: int -> 'a list -> 'a
  val linspan: int -> int -> int -> int list
  val last: 'a list -> 'a
  val intersp: 'a list -> 'a list -> 'a list
  val enum_count_from: int -> 'a list -> int * (int * 'a) list
  val enumerate_from: int -> 'a list -> (int * 'a) list
  val enumerate: 'a list -> (int * 'a) list
  val range: int -> int -> 'a list -> 'a list
  val meta_map: ('a -> 'b) -> 'a list list -> 'b list list
  val meta_maps: ('a -> 'b) -> 'a list list -> 'b list
  val uniques: ''a list -> ''a list
  val log_fold: ('a -> 'b -> 'b) -> 'b -> 'a list -> 'b list
  
  (* Grouping lists *)
  val split_into: int -> 'a list -> 'a list list
  val chop_and_tag: string -> ('a -> string) -> ('a -> bool) -> ('a -> bool) 
    -> 'a list -> (string * 'a list) list
  val bifurcate_at: ''a -> ''a list -> ''a list * ''a * ''a list
  val segmentate_by: ''a list -> ''a list -> ''a list list
  val split_at: string -> string -> string list
  val split_without: string -> string -> string list
  val replace_all: string -> string -> string -> string
  val safe_unprefix: string -> string -> string
  val safe_unsufix: string -> string -> string
  val unprefix_unsufix: string -> string -> string

  (* System operations *)
  val apply_with_timeout: Time.time -> ('a -> 'b) -> 'a -> 'b Exn.result
  val path_prefix_up_to: string -> Path.T -> Path.T option
  val file_exists: string -> bool
  val is_directory: string -> bool
  val create_directory: {force: bool} -> string -> unit
  val create_file: {force: bool} -> string -> string -> string -> unit
  val read_file: string -> string
end;


structure Ops: OPS =
struct

(** GENERIC OPERATIONS **)

fun ith i xs = (if i>=0 then nth xs i else nth (rev xs) (~i - 1));

fun linspan i a b =
  if b < a then 
    raise Fail ("Ops.linspan: " ^ Value.print_int a ^ " should be less than " ^ Value.print_int b)
  else let
    fun add_next [] = add_next [a]
      | add_next (x :: xs) = let val next = x + i
          in if next > b then x :: xs else add_next (next :: x :: xs) end
  in rev (add_next []) end;

fun last [] = raise List.Empty
  | last [x] = x
  | last (_ :: xs) = last xs;

fun intersp seps (x :: (xs as  _ :: _)) = (x :: seps) @ (intersp seps xs)
  | intersp _ xs = xs;

fun enum_count_from _ [] = (0, [])
  | enum_count_from n (x :: xs) = 
    let 
      val (m, ys) = enum_count_from (n + 1) xs;
    in (m+1, (n, x) :: ys) end;

fun enumerate_from _ [] = []
  | enumerate_from n (y::ys) = (n, y) :: (enumerate_from (n+1) ys);

fun enumerate xs = enumerate_from 0 xs;

fun range _ _ [] = []
  | range m n (x :: xs) = 
    if 0 <= m andalso m <= n andalso m <= length (x :: xs) then
    let
      fun process (so_far, count, []) = (so_far, count, [])
        | process (so_far, count, y :: ys) = 
          if count > n then
            (so_far, count, [])
          else if m <= count andalso count <= n then
            process (y::so_far, count+1, ys)
          else   
            process (so_far, count+1, ys)
    in rev (#1 (process ([], 0, x :: xs))) end
    else raise Fail ("Ops.range: out of range input.")

fun meta_map _ [] = []
  | meta_map f xss = rev (fold (fn xs => fn yss => (map f xs) :: yss) xss []);

fun meta_maps f xss = flat (meta_map f xss);

fun uniques xs = Library.distinct (op =) xs;

fun log_fold _ _ [] = []
  | log_fold f s (x :: xs) = let val v = f x s in v :: log_fold f v xs end;


(** GROUPING LISTS **)

fun split_into 1 xs = [xs]
  | split_into n xs = 
  let 
    val total = length xs;
  in if total >= n andalso n > 1 then
    let
      val batch_size = total div n
      fun do_next x (m, ys, batches, count) =
        if m > 0 then (m-1, x::ys, batches, count)
        else if count < n then (batch_size-1, [x], (rev ys) :: batches, count+1)
        else (batch_size, x::ys, batches, count)
      val (_, remainder, rev_batches, _) = fold do_next xs (batch_size, [], [], 1);
      in rev (rev remainder :: rev_batches) end
  else raise Fail "Ops.split_into: non-positive split size or larger than list" end;

fun chop_and_tag tag get_text is_start is_end xs =
  let
    fun add_one x =
      if is_start x then ([()], [(tag, [x])])
      else ([], [("other", [x])]);
    (* lhs is unit stack indicating whether a block is still open *)
    fun add_new x (_, []) = add_one x
      | add_new x (buf, (name, chunks) :: rest) =
        if not (null buf) then
          if is_start x then
            (() :: buf, (name, x :: chunks) :: rest)
          else if is_end x then
            (tl buf, (name, x :: chunks) :: rest)
          else
            (buf, (name, x :: chunks) :: rest)
        else 
          if is_start x then
            (() :: buf, (tag, [x]) :: (name, chunks) :: rest)
          else if is_end x then
            raise Fail ("unopened "^ tag ^" for: " ^ (get_text x))
          else if name = "other" then
            (buf, (name, x :: chunks) :: rest)
          else ([], ("other", [x]) :: (name, chunks) :: rest);
     val rev_result = fold add_new xs ([], []);
     val _ = if not (null (fst rev_result)) then raise Fail ("unbalanced" ^ tag) else ();
   in rev (map (fn (name, chunk) => (name, rev chunk)) (snd rev_result)) end;

fun bifurcate_at a [] = ([], a, [])
  | bifurcate_at a (c :: cs) = 
      if a = c then ([], a, cs)
      else let 
        val (xs, _, ys) = bifurcate_at a cs
      in (c::xs, a, ys) end;

fun segmentate_by [] ys = [ys]
  | segmentate_by _ [] = [] 
  | segmentate_by (x :: xs) (y :: ys) =
     if length xs > length ys then [y :: ys] else
     let 
       val (not_x, _, after_x) = bifurcate_at x (y :: ys);
       (* out = (is_prefix, largest_coinc, rest_of_c_cs) *)
       fun largest_coincide [] cs = (true, [], cs)
         | largest_coincide _ [] = (false, [], [])
         | largest_coincide (d::ds) (c::cs) =
             if not (d = c) then (false, [], c::cs)
             else 
               let
                 val (matched, ds', cs') = largest_coincide ds cs;
                 val result = if matched 
                   then (true, d::ds', cs')
                   else (false, d::ds', cs')
               in result end;
       val (matched, coincidences, rest) = if null after_x 
         then (false, [], []) 
         else largest_coincide (x::xs) (x::after_x);
       val result = if matched 
         then if null not_x 
           then [coincidences] @ segmentate_by (x::xs) rest 
           else [not_x, coincidences] @ segmentate_by (x::xs) rest 
         else let val xss = segmentate_by (x::xs) rest; in 
           if null xss then [not_x @ coincidences] 
           else [not_x @ coincidences @ hd xss] @ (tl xss) end;
     in result end;

fun split_at split_str target_str = 
  segmentate_by (String.explode split_str) (String.explode target_str)
  |> map String.implode;

fun split_without split_str target_str = 
  split_at split_str target_str
  |> filter_out (fn s => Pred.is split_str s);

fun replace_all str1 str2 txt =
  if String.size str1 = 0 then txt
  else let
    fun rec_replace str1 str2 txt =
      let
        val len1 = String.size str1;
        val len_txt = String.size txt;
        fun extr (pos, opt) = String.extract (txt, pos, opt);
        fun find_and_replace pos acc =
          if pos >= len_txt then acc ^ (extr (pos, NONE))
          else (case String.isPrefix str1 (extr (pos, NONE)) of
            true => find_and_replace (pos + len1) (acc ^ str2)
            | false => find_and_replace (pos + 1) (acc ^ (extr (pos,(SOME 1))))
          )
       in find_and_replace 0 "" end
  in rec_replace str1 str2 txt end;

fun safe_unprefix substr target_str =
  if Pred.starts_with substr target_str 
  then Library.unprefix substr target_str 
  else target_str;

fun safe_unsufix substr target_str =
  if Pred.ends_with substr target_str 
  then Library.unsuffix substr target_str 
  else target_str;

fun unprefix_unsufix substr target_str =
  safe_unprefix substr target_str
  |> (fn mid_str => safe_unsufix substr mid_str);


(** SYSTEM OPERATIONS **)

(* WARNING: potential slowdown due to overhead if too many calls to this method *)
(* fun apply_with_timeout timeout f x  = 
  let
    val future_val = Future.fork (fn () => f x)

    fun wait_for_future start_time =
      let
        val curr_time = Time.now ();
        val so_far = curr_time - start_time;
      in if Future.is_finished future_val 
          then Future.get_result future_val
          else if so_far > timeout then (Future.cancel future_val; Exn.capture_body (fn () => raise Timeout.TIMEOUT so_far))
          else (OS.Process.sleep (Time.fromMilliseconds 100); wait_for_future start_time)
      end;
  in wait_for_future (Time.now ()) end; *)

fun apply_with_timeout timeout f x = Exn.result (Timeout.apply timeout f) x

fun path_prefix_up_to parent_name path =
  let
    val full_path = File.absolute_path path;
    fun check_else_parent_path opt_p = (case opt_p of 
      SOME p => 
        if File.platform_path (Path.base p) = parent_name 
        then SOME p 
        else (case try Path.dir p of 
          SOME q => if can Path.dir q 
            then check_else_parent_path (SOME q) 
            else NONE
          | NONE => NONE)
      | NONE => NONE)
  in check_else_parent_path (SOME full_path) end;

fun file_exists file_dir 
  = OS.FileSys.access (file_dir, [OS.FileSys.A_READ]) handle OS.SysErr _ => false;

fun is_directory dir 
  = OS.FileSys.isDir dir handle OS.SysErr _ => false;

fun create_directory {force=forced} dir =
  if is_directory dir then ()
  else let
    val dirs = map (String.implode) (Pred.split_whenL (fn c => str c = "/") (String.explode dir));
    val _ = if null dirs orelse not (hd dirs = "/") 
      then raise Fail "Ops.create_directory: only accepts absolute paths starting with \"/\"" 
      else ()
    fun make dir_str = OS.FileSys.mkDir dir_str
      handle OS.SysErr (cause, err) => (
        raise (Fail ("Ops.create_directory: could not create directory:\n")); 
        Exn.reraise (OS.SysErr (cause, err))
      );
    fun iter str dir_so_far = 
      let
        val new_dir = dir_so_far ^ str;
        val result = if is_directory new_dir orelse not forced then new_dir
          else let val _ = make new_dir in new_dir end;
      in result end;
    val _ = fold iter dirs "";
  in () end;

fun create_file {force=forced} dir name content =
  let
    val _ = create_directory {force=forced} dir
        handle Fail txt => Exn.reraise (Fail txt);
    val filepath = OS.Path.joinDirFile {dir=dir, file=name};
  in
    if file_exists filepath andalso not forced then () else
      let val outstream = TextIO.openOut filepath; in
        TextIO.output (outstream, content);
        TextIO.closeOut outstream end
  end;

fun read_file fname = 
  let 
    val IO_instream = TextIO.openIn fname;
    val err = Exn.result TextIO.inputAll IO_instream;
    val _ = TextIO.closeIn IO_instream;
    val result = Exn.release err
  in result end;


end;
