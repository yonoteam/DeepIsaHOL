
signature OPS =
sig
  (* Generic operations *)
  val ith: int -> 'a list -> 'a
  val last: 'a list -> 'a
  val insert_if: ('a -> bool) -> 'a -> 'a list -> 'a list
  val intersp: 'a list -> 'a list -> 'a list
  val enumerate_from: int -> 'a list -> (int * 'a) list
  val enumerate: 'a list -> (int * 'a) list
  val uniques: ''a list -> ''a list
  val log_fold: ('a -> 'b -> 'b) -> 'b -> 'a list -> 'b list
  val chop_and_tag: string -> ('a -> string) -> ('a -> bool) -> ('a -> bool) 
    -> 'a list -> (string * 'a list) list
  val bifurcate_at: ''a -> ''a list -> ''a list * ''a * ''a list
  val segmentate_by: ''a list -> ''a list -> ''a list list
  val split_at: string -> string -> string list
  val replace_all: string -> string -> string -> string

  (* IO operations *)
  val file_exists: string -> bool
  val is_directory: string -> bool
  val create_directory: {force: bool} -> string -> unit
  val create_file: {force: bool} -> string -> string -> string -> unit
  val read_file: string -> string
end;


structure Ops: OPS =
struct

(** GENERIC OPERATIONS **)

(* ith: int -> 'a list -> 'a *)
fun ith i xs = (if i>=0 then nth xs i else nth (rev xs) (~i - 1));

(* last: 'a list -> 'a *)
fun last [] = raise List.Empty
  | last [x] = x
  | last (_ :: xs) = last xs;

(* insert_if: ('a -> bool) -> 'a -> 'a list -> 'a list *)
fun insert_if P x xs = if P x then x :: xs else xs;

(* intersp: 'a list -> 'a list -> 'a list *)
fun intersp seps (x :: (xs as  _ :: _)) = (x :: seps) @ (intersp seps xs)
  | intersp _ xs = xs;

(* enumerate_from: int -> 'a list -> (int * 'a) list *)
fun enumerate_from _ [] = []
  | enumerate_from n (y::ys) = (n, y) :: (enumerate_from (n+1) ys);

(* enumerate: 'a list -> (int * 'a) list *)
fun enumerate xs = enumerate_from 0 xs;

(* uniques: 'a list -> 'a list *)
fun uniques xs = Library.distinct (op =) xs;

(* log_fold: ('a -> 'b -> 'b) -> 'b -> 'a list -> 'b list *)
fun log_fold _ _ [] = []
  | log_fold f s (x :: xs) = let val v = f x s in v :: log_fold f v xs end;

(* chop_and_tag: string -> ('a -> string) -> ('a -> bool) -> ('a -> bool) 
  -> 'a list -> (string * 'a list) list *)
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

(* bifurcate_at: ''a -> ''a list -> ''a list * ''a * ''a list *)
fun bifurcate_at a [] = ([], a, [])
  | bifurcate_at a (c :: cs) = 
      if a = c then ([], a, cs)
      else let 
        val (xs, _, ys) = bifurcate_at a cs
      in (c::xs, a, ys) end;

(* segmentate_by: ''a list -> ''a list -> ''a list list *)
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

(* split_at: string -> string -> string list *)
fun split_at split_str target_str = space_explode split_str target_str;

(* replace_all: string -> string -> string -> string *)
fun replace_all str1 str2 txt =
  let
    val chars = String.explode txt;
    val cs1 = String.explode str1;
    val cs2 = String.explode str2;
    fun replace subcs = if subcs = cs1 then cs2 else subcs;
  in String.concat (map (String.implode o replace) (segmentate_by cs1 chars)) end;


(** IO OPERATIONS **)

(* file_exists: string -> bool *)
fun file_exists file_dir 
  = OS.FileSys.access (file_dir, [OS.FileSys.A_READ]) handle OS.SysErr _ => false;

(* is_directory: string -> bool *)
fun is_directory dir 
  = OS.FileSys.isDir dir handle OS.SysErr _ => false;

(* create_directory: {force: bool} -> string -> unit *)
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

(* create_file: {force: bool} -> string -> string -> string -> unit *)
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

(* read_file: string -> string *)
fun read_file fname = 
  let 
    val IO_instream = TextIO.openIn fname;
    val content = TextIO.inputAll IO_instream;
    val _ = TextIO.closeIn IO_instream;
  in content end;


end;
