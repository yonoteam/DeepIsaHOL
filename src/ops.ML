
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
  val split_by: string -> string -> string list

  (* IO operations *)
  val file_exists: string -> bool
  val is_directory: string -> bool
  val create_file: bool -> string -> string -> string -> unit
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

(* split_by: string -> string -> string list *)
fun split_by split_str target_str = space_explode split_str target_str;


(** IO OPERATIONS **)

(* string -> bool *)
fun file_exists file_dir 
  = OS.FileSys.access (file_dir, [OS.FileSys.A_READ]) handle OS.SysErr _ => false;

(* string -> bool *)
fun is_directory dir 
  = OS.FileSys.isDir dir handle OS.SysErr _ => false;

(* bool -> string -> string -> string -> unit *)
fun create_file force dir name content =
  let
    val _ = if is_directory dir then () else OS.FileSys.mkDir dir;
    val filepath = OS.Path.joinDirFile {dir=dir, file=name};
  in
    if file_exists filepath andalso not force then () else
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
